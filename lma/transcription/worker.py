from queue import Empty, Queue

from lma.schemas import AudioChunk, TranscriptChunk
from lma.transcription.whisper import WhisperTranscriber
from lma.workers.base import Worker

class TranscriptionWorker(Worker):

    def __init__(
        self,
        input_queue: Queue[AudioChunk],
        output_queue: Queue[TranscriptChunk],
        transcriber: WhisperTranscriber,
    ) -> None:

        super().__init__(
            name="transcription-worker"
        )

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.transcriber = transcriber
    
    
    def run(self) -> None:

        while not self.stopped:

            try:
                chunk = self.input_queue.get(
                    timeout=0.5
                )

            except Empty:
                continue

            try:
                transcript = self.transcriber.transcribe(
                    chunk
                )

                self.output_queue.put(
                    transcript
                )

            finally:
                self.input_queue.task_done()