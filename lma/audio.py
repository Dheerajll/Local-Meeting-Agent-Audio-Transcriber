"""
Audio capture + VAD-based chunking.

Captures system audio via ffmpeg, runs it through Silero VAD, and publishes
finished chunks as AudioChunk objects through an injected ChunkPublisher.

This module never touches the filesystem and knows nothing about Whisper,
storage, or the backend — it only produces AudioChunk objects and hands them
off. What happens to a chunk after that is someone else's problem.
"""

import subprocess

import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

from .audio_device import AudioDeviceManager
from .buffer import RollingAudioBuffer
from .constants import (
    CHANNELS,
    ChunkReason,
    FINALIZE_SILENCE_MS,
    FRAME_BYTES,
    HARD_LIMIT_MS,
    OVERLAP_MS,
    RecorderState,
    RESUME_WINDOW_MS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    SOFT_LIMIT_MS,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
)
from .exceptions import AudioDeviceError, FFmpegError
from .publisher import ChunkPublisher
from .schemas import AudioChunk

torch.set_num_threads(1)

BYTES_PER_MS = SAMPLE_RATE * SAMPLE_WIDTH // 1000
OVERLAP_BYTES = OVERLAP_MS * BYTES_PER_MS
SOFT_LIMIT_BYTES = SOFT_LIMIT_MS * BYTES_PER_MS
HARD_LIMIT_BYTES = HARD_LIMIT_MS * BYTES_PER_MS
RESUME_WINDOW_BYTES = RESUME_WINDOW_MS * BYTES_PER_MS
FINALIZE_SILENCE_BYTES = FINALIZE_SILENCE_MS * BYTES_PER_MS


class AudioRecorder:
    """
    State machine:

        IDLE
            -> RECORDING            speech detected
        RECORDING
            -> WAITING_FOR_RESUME   silence >= RESUME_WINDOW_MS
        WAITING_FOR_RESUME
            -> RECORDING            speech resumes
            -> (finalize) -> IDLE   silence >= FINALIZE_SILENCE_MS
        RECORDING / WAITING_FOR_RESUME
            -> (finalize) -> IDLE   buffer >= HARD_LIMIT_MS (forced)

    Crossing SOFT_LIMIT_MS doesn't change the state or shorten the wait for
    silence — it only changes which ChunkReason a natural-silence finalize
    gets tagged with, so downstream consumers can tell "this chunk closed
    normally" from "this chunk closed normally, but it had already grown
    past the soft limit first".
    """

    def __init__(self, publisher: ChunkPublisher[AudioChunk], vad: VADIterator | None = None):
        self.publisher = publisher
        self.device_manager = AudioDeviceManager()

        self.process = None
        self.running = False
        self.leftover = bytearray()

        self.speech_buffer = RollingAudioBuffer()
        self.overlap_buffer = RollingAudioBuffer()

        self.state = RecorderState.IDLE
        self.is_speaking = False
        self.last_speech_offset = 0
        self.over_soft_limit = False

        self.chunk_id = 0
        self.chunk_start_ms = 0
        self.chunk_overlap_ms = 0

        # vad is injectable so tests/other callers can pass a stub instead
        # of pulling down the real Silero model — same DI principle as the
        # publisher.
        if vad is None:
            model = load_silero_vad()
            vad = VADIterator(
                model,
                threshold=VAD_THRESHOLD,
                sampling_rate=SAMPLE_RATE,
                min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                speech_pad_ms=VAD_SPEECH_PAD_MS,
            )
        self.vad = vad

    # ---------- lifecycle ----------

    def start(self) -> None:
        try:
            self.device_manager.start_recording_mode()
        except Exception as exc:
            raise AudioDeviceError(f"Could not switch to recording device: {exc}") from exc

        cmd = [
            "ffmpeg",
            "-f", "avfoundation",
            "-i", ":0",
            "-ac", str(CHANNELS),
            "-ar", str(SAMPLE_RATE),
            "-f", "s16le",
            "pipe:1",
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=4096,
            )
        except OSError as exc:
            raise FFmpegError(f"Failed to start ffmpeg: {exc}") from exc

        self.running = True
        self.loop()

    def loop(self) -> None:
        while self.running:
            raw = self.process.stdout.read(4096)
            if not raw:
                break
            self.leftover.extend(raw)

            while len(self.leftover) >= FRAME_BYTES:
                pcm = bytes(self.leftover[:FRAME_BYTES])
                del self.leftover[:FRAME_BYTES]
                self.process_frame(pcm)

    def stop(self) -> None:
        self.running = False
        if self.process:
            self.process.terminate()

        # Flush whatever's in progress instead of dropping it. No dedicated
        # "stream ended" reason in ChunkReason, so this reuses HARD_LIMIT —
        # it's a forced cut, just not a duration-triggered one.
        if self.state != RecorderState.IDLE and len(self.speech_buffer):
            self._finalize(reason=ChunkReason.HARD_LIMIT, forced=True)

        try:
            self.device_manager.stop_recording_mode()
        except Exception as exc:
            raise AudioDeviceError(f"Could not restore original audio device: {exc}") from exc

    # ---------- per-frame processing ----------

    def process_frame(self, pcm: bytes) -> None:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        result = self.vad(torch.from_numpy(audio), return_seconds=False)

        if result:
            if "start" in result:
                self.is_speaking = True
                print("🎙 VAD: speech detected")
            elif "end" in result:
                self.is_speaking = False
                print("🤫 VAD: silence detected")

        if self.state == RecorderState.IDLE:
            if self.is_speaking:
                self._begin_chunk()
            else:
                return  # nothing to record yet

        self.speech_buffer.append(pcm)
        if self.is_speaking:
            self.last_speech_offset = len(self.speech_buffer)

        self._update_state()

    def _begin_chunk(self) -> None:
        self.state = RecorderState.RECORDING
        self.over_soft_limit = False
        seed = self.overlap_buffer.read()
        self.speech_buffer = RollingAudioBuffer(seed)
        self.last_speech_offset = len(self.speech_buffer)
        self.chunk_overlap_ms = len(seed) // BYTES_PER_MS
        print(f"🗣 Chunk {self.chunk_id + 1} started (seeded with {self.chunk_overlap_ms}ms overlap)")

    def _update_state(self) -> None:
        current = len(self.speech_buffer)

        # Hard limit always wins, speaking or not.
        if current >= HARD_LIMIT_BYTES:
            self._finalize(reason=ChunkReason.HARD_LIMIT, forced=True)
            return

        if current >= SOFT_LIMIT_BYTES:
            if not self.over_soft_limit:
                print(f"⏱ Soft limit reached ({current // BYTES_PER_MS}ms) — now watching for a pause to close on")
            self.over_soft_limit = True

        if self.is_speaking:
            self.state = RecorderState.RECORDING
            return

        silence_bytes = current - self.last_speech_offset

        if silence_bytes >= FINALIZE_SILENCE_BYTES:
            reason = ChunkReason.SOFT_LIMIT if self.over_soft_limit else ChunkReason.NATURAL_SILENCE
            self._finalize(reason=reason, forced=False)
        elif silence_bytes >= RESUME_WINDOW_BYTES:
            if self.state != RecorderState.WAITING_FOR_RESUME:
                print(f"⏳ {silence_bytes // BYTES_PER_MS}ms silence — waiting to see if speaker resumes")
            self.state = RecorderState.WAITING_FOR_RESUME
        # else: short pause — stay in RECORDING, keep buffering through it

    # ---------- finalize ----------

    def _finalize(self, reason: ChunkReason, forced: bool) -> None:
        if not len(self.speech_buffer):
            self.state = RecorderState.IDLE
            return

        self.chunk_id += 1
        pcm_bytes = self.speech_buffer.read()
        duration_ms = len(pcm_bytes) // BYTES_PER_MS
        end_ms = self.chunk_start_ms + duration_ms

        print(
        f"💾 Chunk {self.chunk_id} published | reason={reason.value} "
        f"forced={forced} duration={duration_ms}ms "
        f"[{self.chunk_start_ms}ms -> {end_ms}ms]"
    )

        chunk = AudioChunk(
            chunk_id=self.chunk_id,
            pcm_bytes=pcm_bytes,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            start_ms=self.chunk_start_ms,
            end_ms=end_ms,
            overlap_ms=self.chunk_overlap_ms,
            reason=reason,
            forced=forced,
        )
        self.publisher.publish(chunk)

        overlap_pcm = self.speech_buffer.tail(OVERLAP_BYTES)
        overlap_ms_actual = len(overlap_pcm) // BYTES_PER_MS
        self.overlap_buffer = RollingAudioBuffer(overlap_pcm)

        self.speech_buffer.clear()
        self.chunk_start_ms = end_ms - overlap_ms_actual
        self.last_speech_offset = 0
        self.over_soft_limit = False
        self.state = RecorderState.IDLE

        # A hard-limit cut can land mid-speech. is_speaking is still True in
        # that case (VAD won't refire "start" for a segment that never
        # stopped), so resume immediately instead of waiting for an event
        # that isn't coming.
        if self.is_speaking:
            self._begin_chunk()