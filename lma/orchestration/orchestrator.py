"""
MeetingOrchestrator — the top-level coordinator.
No separate monitor thread. Meeting-end detection runs
inside the main audio loop to stay thread-safe with Playwright.
"""
import warnings
# Suppress Pyannote/PyTorch standard deviation warnings on short audio chunks
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

import queue
import threading
import time
from urllib.parse import urlparse

from lma.core.constants import RecorderState, ChunkReason, FRAME_DURATION_MS
from lma.core.schemas import SessionPaths
from lma.core.paths import create_session_dir

from lma.audio.capture import FFmpegCapture
from lma.audio.clock import AudioClock
from lma.audio.source import AudioSource
from lma.audio.recorder import AudioRecorder
from lma.audio.detector.silero import SileroDetector
from lma.audio.chunk_buffer import ChunkBuffer
from lma.audio.chunk_builder import ChunkBuilder
from lma.audio.device import AudioDeviceManager
from lma.audio.wav_writer import WavWriter

from lma.transcription.whisper_transcriber import WhisperTranscriber
from lma.transcription.transcript_writer import TranscriptWriter
from lma.transcription.worker import TranscriptionWorker
from lma.transcription.diarizer import Diarizer
from lma.transcription.speaker_manager import SpeakerManager
from lma.transcription.word_aligner import WordAligner
from lma.transcription.cache import get_model_snapshot_path

from lma.browser.manager import BrowserManager
from lma.workers.publisher import ChunkPublisher

# Configuration
BLACKHOLE_INPUT_DEVICE = "BlackHole 2ch"
RECORDING_OUTPUT_DEVICE = "Local meeting agent output"
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

# How often to check if the meeting has ended (in ms of audio time)
MEETING_CHECK_INTERVAL_MS = 5_000

# Don't check for the first N seconds after joining
MEETING_CHECK_GRACE_MS = 15_000


class MeetingOrchestrator:
    def __init__(
        self,
        meeting_url: str,
        session_id: str | None = None,
        language: str | None = None,
    ):
        self.meeting_url = meeting_url
        self.language = language
        self.session_id = session_id

        self.session_paths: SessionPaths | None = None
        self.browser: BrowserManager | None = None
        self.audio_device_manager: AudioDeviceManager | None = None
        self.worker: TranscriptionWorker | None = None
        self.source: AudioSource | None = None
        self._recorder: AudioRecorder | None = None

        self.chunk_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        # Track why the meeting ended (for the summary)
        self._end_reason: str = ""

    # ================================================================
    # PUBLIC API
    # ================================================================

    def run(self) -> None:
        try:
            self._setup()
            self._join_meeting()
            self._start_audio_pipeline()
            self._main_loop()
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user.")
        except Exception as exc:
            print(f"\n❌ Orchestrator error: {exc}")
            raise
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._stop_event.set()

    # ================================================================
    # SETUP
    # ================================================================

    def _setup(self) -> None:
        print("📁 Creating session...")
        self.session_paths = create_session_dir(
            session_id=self.session_id,
        )
        print(f"   Session: {self.session_paths.root}")

    # ================================================================
    # BROWSER
    # ================================================================

    def _join_meeting(self) -> None:
        print("🌐 Launching browser...")
        self.browser = BrowserManager()
        self.browser.start()

        print(f"🔗 Joining meeting: {self.meeting_url}")
        self.browser.join_meeting(self.meeting_url)
        print("✓ Joined meeting")

    # ================================================================
    # AUDIO PIPELINE
    # ================================================================

    def _start_audio_pipeline(self) -> None:
        # 1. Switch system output to BlackHole aggregate
        print("🔊 Switching audio output...")
        self.audio_device_manager = AudioDeviceManager()
        self.audio_device_manager.start_recording_mode(
            recording_device=RECORDING_OUTPUT_DEVICE,
        )
        time.sleep(1.5)

        # 2. Build transcription components
        print("🤖 Loading AI models...")
        model_path = get_model_snapshot_path(WHISPER_MODEL)

        transcriber = WhisperTranscriber(
            model_path=model_path,
            diarizer=Diarizer(),
            speaker_manager=SpeakerManager(similarity_threshold=0.70),
            word_aligner=WordAligner(),
            language=self.language,
        )

        wav_writer = WavWriter(self.session_paths.audio)
        transcript_writer = TranscriptWriter(
            self.session_paths.transcripts,
        )

        # 3. Start transcription worker thread
        print("🧵 Starting transcription worker...")
        self.worker = TranscriptionWorker(
            input_queue=self.chunk_queue,
            wav_writer=wav_writer,
            transcriber=transcriber,
            transcript_writer=transcript_writer,
        )
        self.worker.start()

        # 4. Build audio recorder
        publisher = ChunkPublisher(self.chunk_queue)
        self._recorder = AudioRecorder(
            detector=SileroDetector(),
            chunk_buffer=ChunkBuffer(),
            chunk_builder=ChunkBuilder(),
            publisher=publisher,
        )

        # 5. Start FFmpeg capture from BlackHole
        print(f"🎙️  Starting audio capture: {BLACKHOLE_INPUT_DEVICE}")
        capture = FFmpegCapture(device=BLACKHOLE_INPUT_DEVICE)
        clock = AudioClock()
        self.source = AudioSource(capture, clock)
        self.source.start()

        print("✓ Audio pipeline running")

    # ================================================================
    # MAIN LOOP  (meeting detection is inline, no separate thread)
    # ================================================================

    def _main_loop(self) -> None:
        print("🎙️  Recording... (waiting for speech)")
        print(
            "   Meeting-end detection runs every "
            f"{MEETING_CHECK_INTERVAL_MS // 1000}s.\n"
        )

        last_check_ms = 0

        for frame in self.source.frames():
            # External stop signal
            if self._stop_event.is_set():
                print("\n⏹️  Stop signal received.")
                self._end_reason = "stop_signal"
                break

            # Process audio through VAD + chunking
            self._recorder.process(frame)

            # Periodically check if the meeting has ended.
            # We use the audio clock (frame.timestamp_ms) instead of
            # wall-clock time so checks stay in sync with the stream.
            if frame.timestamp_ms >= MEETING_CHECK_GRACE_MS:
                elapsed = frame.timestamp_ms - last_check_ms
                if elapsed >= MEETING_CHECK_INTERVAL_MS:
                    last_check_ms = frame.timestamp_ms
                    if self._is_meeting_over():
                        print(
                            "\n🔴 Meeting ended "
                            f"({self._end_reason})."
                        )
                        break

        # Force-finalize any in-progress chunk
        self._finalize_remaining()

    # ================================================================
    # MEETING-END DETECTION  (runs in the main thread — thread-safe)
    # ================================================================

    def _is_meeting_over(self) -> bool:
        """
        Check the browser page to see if the meeting has ended.
        All Playwright calls happen in the main thread, so this
        is safe with the sync API.
        """
        try:
            page = self.browser.page

            # 1. Is the page still alive?
            try:
                current_url = page.url
            except Exception:
                self._end_reason = "page_closed"
                return True

            # 2. Did the URL navigate away from the meeting?
            if self._url_changed(current_url):
                self._end_reason = "url_changed"
                return True

            # 3. Google Meet: "You left the meeting" screen
            if "meet.google.com" in current_url:
                if self._check_text_visible(
                    page, "You left the meeting"
                ):
                    self._end_reason = "left_meeting"
                    return True

                if self._check_button_visible(page, "Rejoin"):
                    self._end_reason = "left_meeting"
                    return True

            # 4. Generic "meeting ended" indicators
            ended_phrases = [
                "This meeting has ended",
                "The meeting has ended",
                "Meeting ended",
                "You have been removed",
            ]
            for phrase in ended_phrases:
                if self._check_text_visible(page, phrase):
                    self._end_reason = "meeting_ended"
                    return True

            return False

        except Exception:
            # Any unexpected error means the page is gone
            self._end_reason = "page_error"
            return True

    def _url_changed(self, current_url: str) -> bool:
        original = urlparse(self.meeting_url)
        current = urlparse(current_url)

        if original.hostname != current.hostname:
            return True
        if original.path.strip("/") and not current.path.strip("/"):
            return True
        return False

    @staticmethod
    def _check_text_visible(page, text: str) -> bool:
        try:
            return page.get_by_text(
                text, exact=False
            ).count() > 0
        except Exception:
            return False

    @staticmethod
    def _check_button_visible(page, name: str) -> bool:
        try:
            return page.get_by_role(
                "button", name=name
            ).count() > 0
        except Exception:
            return False

    # ================================================================
    # FINALIZE
    # ================================================================

    def _finalize_remaining(self) -> None:
        if (
            self._recorder
            and self._recorder.state != RecorderState.IDLE
        ):
            print("📦 Force-finalizing last chunk...")
            self._recorder._finalize(
                reason=ChunkReason.STREAM_ENDED,
                forced=True,
            )

    # ================================================================
    # SHUTDOWN
    # ================================================================

    def _shutdown(self) -> None:
        print("\n🧹 Shutting down...")

        # Phase 1: Stop producing new data
        if self.source is not None:
            print("   Stopping audio capture...")
            self.source.stop()

        # Phase 2: Give the user their system back immediately
        if self.audio_device_manager is not None:
            print("   Restoring audio device...")
            self.audio_device_manager.stop_recording_mode()

        if self.browser is not None:
            print("   Closing browser...")
            self.browser.close()

        # Phase 3: Wait for transcription to finish
        pending = self.chunk_queue.qsize()
        if pending > 0:
            print(
                f"   ⏳ Waiting for {pending} chunk(s) "
                f"to transcribe..."
            )

        self.chunk_queue.join()
        print("   ✓ All chunks transcribed")

        # Phase 4: Stop the worker
        if self.worker is not None:
            print("   Stopping transcription worker...")
            self.worker.stop()

        # Phase 5: Summary
        self._print_summary()
        print("\n✅ Orchestrator shutdown complete.")

    def _print_summary(self) -> None:
        if self.session_paths is None:
            return

        transcript_files = sorted(
            self.session_paths.transcripts.glob("chunk_*.json"),
        )

        print(f"\n{'=' * 60}")
        print(f"📋 SESSION SUMMARY")
        print(f"{'=' * 60}")
        print(f"   Meeting URL:  {self.meeting_url}")
        print(f"   Session dir:  {self.session_paths.root}")
        print(f"   Chunks saved: {len(transcript_files)}")
        if self._end_reason:
            print(f"   End reason:   {self._end_reason}")
        print(f"{'=' * 60}\n")