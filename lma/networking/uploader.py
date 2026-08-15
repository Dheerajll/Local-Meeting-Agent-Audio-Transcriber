"""
ChunkUploader — bridges the TranscriptionWorker to the BackendClient.

When a transcript chunk is generated, the uploader sends it
to the backend via WebSocket. Falls back to local-only mode
if the backend is not configured or not reachable.
"""

from lma.core.schemas import TranscriptChunk
from lma.core.config import is_configured
from lma.networking.client import BackendClient


class ChunkUploader:
    """
    Optional component that can be injected into TranscriptionWorker.

    If the backend is configured and reachable, chunks are streamed
    in real-time. Otherwise, chunks are only saved locally.
    """

    def __init__(self, meeting_id: str | None = None):
        self.meeting_id = meeting_id
        self.client: BackendClient | None = None
        self._enabled = False

    def start(self) -> None:
        """
        Start the uploader. Only connects if the LMA is configured
        with a token and backend URL.
        """
        if not is_configured():
            print("ℹ️  Backend not configured. Running in local-only mode.")
            return

        if not self.meeting_id:
            print("ℹ️  No meeting ID. Running in local-only mode.")
            return

        try:
            self.client = BackendClient(meeting_id=self.meeting_id)
            self.client.start()
            self._enabled = True
            print("✓ Chunk uploader started (backend streaming enabled)")
        except Exception as exc:
            print(f"⚠️  Could not start uploader: {exc}")
            print("   Running in local-only mode.")

    def upload(self, chunk: TranscriptChunk) -> None:
        """
        Send a transcript chunk to the backend.
        Called by TranscriptionWorker after saving locally.
        """
        if not self._enabled or self.client is None:
            return

        if not self.client.is_connected:
            print(f"  ⚠️  Backend not connected. Chunk {chunk.chunk_id} saved locally only.")
            return

        self.client.send_chunk(chunk)

    def stop(self) -> None:
        """Stop the uploader and close the WebSocket connection."""
        if self.client is not None:
            self.client.stop()
            self.client = None
        self._enabled = False