"""
MeetingOrchestrator — the top-level coordinator.
"""
import queue
import threading
import time

# Note: Adjust these imports slightly if you kept constants/schemas at the root lma/ level
from lma.core.constants import SAMPLE_RATE, CHANNELS, RecorderState, ChunkReason
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
from lma.browser.monitor import MeetingMonitor
from lma.workers.publisher import ChunkPublisher

# Configuration
BLACKHOLE_INPUT_DEVICE = "BlackHole 2ch"
RECORDING_OUTPUT_DEVICE = "Local meeting agent output"
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

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

        self.session_paths: SessionPaths 
        self.browser: BrowserManager 
        self.audio_device_manager: AudioDeviceManager | None = None
        self.monitor: MeetingMonitor 
        self.worker: TranscriptionWorker | None = None
        self.source: AudioSource 
        self._recorder: AudioRecorder 

        self.chunk_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            self._setup()
            self._join_meeting()
            self._start_audio_pipeline()
            self._start_monitor()
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

    def _setup(self) -> None:
        print("📁 Creating session...")
        self.session_paths = create_session_dir(session_id=self.session_id)
        print(f"   Session: {self.session_paths.root}")

    def _join_meeting(self) -> None:
        print("🌐 Launching browser...")
        self.browser = BrowserManager()
        self.browser.start()
        
        print(f"🔗 Joining meeting: {self.meeting_url}")
        self.browser.join_meeting(self.meeting_url)
        print("✓ Joined meeting")

    def _start_audio_pipeline(self) -> None:
        print("🔊 Switching audio output...")
        self.audio_device_manager = AudioDeviceManager()
        self.audio_device_manager.start_recording_mode(recording_device=RECORDING_OUTPUT_DEVICE)
        time.sleep(1.5) # Let macOS switch devices

        print("🤖 Loading AI models...")
        model_path = get_model_snapshot_path(WHISPER_MODEL)
        
        transcriber = WhisperTranscriber(
            model_path=model_path,
            diarizer=Diarizer(),
            speaker_manager=SpeakerManager(similarity_threshold=0.70),
            word_aligner=WordAligner(),
            language=self.language,
        )

        if self.session_paths:
            wav_writer = WavWriter(self.session_paths.audio)
            transcript_writer = TranscriptWriter(self.session_paths.transcripts)

        print("🧵 Starting transcription worker...")
        self.worker = TranscriptionWorker(
            input_queue=self.chunk_queue,
            wav_writer=wav_writer,
            transcriber=transcriber,
            transcript_writer=transcript_writer,
        )
        self.worker.start()

        publisher = ChunkPublisher(self.chunk_queue)
        self._recorder = AudioRecorder(
            detector=SileroDetector(),
            chunk_buffer=ChunkBuffer(),
            chunk_builder=ChunkBuilder(),
            publisher=publisher,
        )

        print(f"🎙️  Starting audio capture: {BLACKHOLE_INPUT_DEVICE}")
        capture = FFmpegCapture(device=BLACKHOLE_INPUT_DEVICE)
        clock = AudioClock()
        self.source = AudioSource(capture, clock)
        self.source.start()
        
        print("✓ Audio pipeline running")

    def _start_monitor(self) -> None:
        print("👁️  Starting meeting monitor...")
        self.monitor = MeetingMonitor(
            page=self.browser.page,
            original_url=self.meeting_url,
            poll_interval=5.0,
            grace_period=15.0,
        )
        self.monitor.start()

    def _main_loop(self) -> None:
        print("🎙️  Recording... (waiting for speech)")
        print("   The meeting monitor will detect when the meeting ends.\n")

        for frame in self.source.frames():
            if self._stop_event.is_set():
                print("\n⏹️  Stop signal received.")
                break
            if self.monitor.meeting_ended.is_set():
                print("\n🔴 Meeting ended (detected by monitor).")
                break
            
            self._recorder.process(frame)

        self._finalize_remaining()

    def _finalize_remaining(self) -> None:
        if self._recorder and self._recorder.state != RecorderState.IDLE:
            print("📦 Force-finalizing last chunk...")
            self._recorder._finalize(reason=ChunkReason.STREAM_ENDED, forced=True)

    def _shutdown(self) -> None:
        print("\n🧹 Shutting down...")

        if self.source: 
            print("   Stopping audio capture...")
            self.source.stop()
        if self.monitor: 
            print("   Stopping meeting monitor...")
            self.monitor.stop()
            
        print("   Waiting for transcription queue to drain...")
        self.chunk_queue.join()
        
        if self.worker: 
            print("   Stopping transcription worker...")
            self.worker.stop()
        if self.audio_device_manager: 
            print("   Restoring audio device...")
            self.audio_device_manager.stop_recording_mode()
        if self.browser: 
            print("   Closing browser...")
            self.browser.close()

        self._print_summary()
        print("\n✅ Orchestrator shutdown complete.")

    def _print_summary(self) -> None:
        if not self.session_paths: return
        transcript_files = sorted(self.session_paths.transcripts.glob("chunk_*.json"))
        
        print(f"\n{'='*60}")
        print(f"📋 SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"   Meeting URL:  {self.meeting_url}")
        print(f"   Session dir:  {self.session_paths.root}")
        print(f"   Chunks saved: {len(transcript_files)}")
        if self.monitor and self.monitor.result.ended:
            print(f"   End reason:   {self.monitor.result.reason.value}")
        print(f"{'='*60}\n")