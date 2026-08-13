from queue import Queue

from lma.core.schemas import (AudioChunk,TranscriptChunk)


transcription_queue: Queue[AudioChunk] = Queue()

backend_queue: Queue[TranscriptChunk] = Queue()