import queue
import wave
import threading

from lma.schemas import AudioChunk


OUTPUT_DIR = "test_chunks"


def save_chunk(chunk: AudioChunk):

    filename = (
        f"{OUTPUT_DIR}/chunk_{chunk.chunk_id:03d}.wav"
    )

    with wave.open(filename, "wb") as wf:

        wf.setnchannels(
            chunk.channels
        )

        wf.setsampwidth(
            2
        )

        wf.setframerate(
            chunk.sample_rate
        )

        wf.writeframes(
            chunk.pcm_bytes
        )

    print(
        f"💾 Saved {filename} "
        f"duration={chunk.end_ms - chunk.start_ms}ms "
        f"reason={chunk.reason.value}"
    )



def chunk_writer_worker(
    chunk_queue: queue.Queue
):

    while True:

        chunk = chunk_queue.get()

        if chunk is None:
            break

        save_chunk(chunk)

        chunk_queue.task_done()