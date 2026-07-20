"""
Consumes AudioChunks from transcription_queue, transcribes them locally,
and publishes TranscriptChunks onto backend_queue.

Runs as a background Worker so it can pull chunks as the recorder produces
them (see the producer/consumer thread setup from a few messages back)
instead of waiting for recording to finish.
"""

from queue import Empty, Queue

from .constants import ChunkReason
from .exceptions import TranscriptionError
from .publisher import ChunkPublisher
from .schemas import AudioChunk, TranscriptChunk
from .transcriber import average_confidence, transcribe_pcm
from .worker import Worker


class TranscriptionWorker(Worker):

    def __init__(
        self,
        queue: "Queue[AudioChunk]",
        publisher: ChunkPublisher,
        get_timeout: float = 0.5,
    ):
        # get_timeout controls how often the loop wakes up to check
        # self.stopped when the queue is empty — small enough that stop()
        # takes effect promptly, large enough not to busy-loop.
        super().__init__(name="TranscriptionWorker")
        self.queue = queue
        self.publisher = publisher
        self._get_timeout = get_timeout

    def run(self) -> None:
        while not self.stopped:
            try:
                audio_chunk = self.queue.get(timeout=self._get_timeout)
            except Empty:
                continue

            try:
                self._process(audio_chunk)
            except Exception as exc:
                # One bad chunk (corrupt audio, model hiccup) shouldn't take
                # the whole pipeline down — log it and keep consuming.
                print(f"⚠ TranscriptionError on chunk {audio_chunk.chunk_id}: {exc}")
            finally:
                self.queue.task_done()

    def _process(self, audio_chunk: AudioChunk) -> None:
        try:
            result = transcribe_pcm(audio_chunk.pcm_bytes, audio_chunk.sample_rate)
        except Exception as exc:
            raise TranscriptionError(str(exc)) from exc

        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        confidence = average_confidence(result)

        transcript_chunk = TranscriptChunk(
            chunk_id=audio_chunk.chunk_id,
            raw_text=text,
            confidence=confidence,
            language=language,
            start_ms=audio_chunk.start_ms,
            end_ms=audio_chunk.end_ms,
            reason=audio_chunk.reason,
            forced=audio_chunk.forced,
            metadata={"overlap_ms": audio_chunk.overlap_ms},
        )

        print(
            f"📝 Chunk {transcript_chunk.chunk_id} transcribed "
            f"({len(text)} chars, confidence={confidence:.2f}): "
            f"{text[:60]!r}{'...' if len(text) > 60 else ''}"
        )

        self.publisher.publish(transcript_chunk)