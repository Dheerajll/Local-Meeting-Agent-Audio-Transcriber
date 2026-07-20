
from .schemas import AudioChunk,TranscriptChunk
from typing import Generic, TypeVar
from queue import Queue

T = TypeVar("T")

class ChunkPublisher(Generic[T]):

    """
    Responsible for publishing chunks.
    Today:
        Queue

    Tomorrow:
        Redis
        RabbitMQ
        WebSocket
    """
    def __init__(self,queue:Queue[T]) -> None:
        self.queue = queue

    def publish(self,chunk: T) -> None:
        self.queue.put(chunk)