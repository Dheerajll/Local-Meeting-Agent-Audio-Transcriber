from pathlib import Path
from queue import Empty,Queue
from threading import Thread, Event

from lma.schemas import AudioChunk

from lma.audio.wav_writer import WavWriter
from lma.transcription.transcriber import WhisperTranscriber
from lma.transcription.transcript_writer import TranscriptWriter


class TranscriptionWorker:
    """
    Consumes AudioChunks and produces persisted TranscriptChunks.

    Pipeline:

        AudioChunk
            ↓
        WavWriter
            ↓
        WhisperTranscriber
            ↓
        TranscriptWriter

    The worker does not know anything about:
        - AudioRecorder
        - backend delivery
        - session cleanup
    """

    def __init__(self,input_queue: Queue[AudioChunk],wav_writer: WavWriter,transcriber: WhisperTranscriber,
                 transcript_writer: TranscriptWriter) -> None:

        self.input_queue = input_queue
        self.wav_writer = wav_writer
        self.transcriber = transcriber
        self.transcript_writer = transcript_writer

        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        """
        Start the transcription worker thread.
        """

        if self._thread is not None:
            raise RuntimeError(
                "TranscriptionWorker is already running."
            )

        self._stop_event.clear()

        self._thread = Thread(
            target=self._run,
            name="transcription-worker",
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> None:
        """
        Request the worker to stop and wait for it.
        """

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

            self._thread = None

    def _run(self) -> None:

        while not self._stop_event.is_set():

            try:
                chunk = self.input_queue.get(
                    timeout=0.1
                )

            except Empty:
                continue

            wav_path: Path | None = None

            try:
                print(
                    f"🎙️ Transcribing chunk "
                    f"{chunk.chunk_id}"
                )

                wav_path = self.wav_writer.write(
                    chunk
                )

                transcript = (
                    self.transcriber.transcribe(
                        wav_path,
                        chunk,
                    )
                )

                path = self.transcript_writer.write(
                    transcript
                )

                print(
                    f"📝 Transcript saved: {path}"
                )

            except Exception as exc:

                print(
                    f"❌ Failed to transcribe "
                    f"chunk {chunk.chunk_id}: {exc}"
                )

            finally:
                '''
                if wav_path is not None:
                    self.wav_writer.delete(
                        wav_path
                    )
                '''

                self.input_queue.task_done()