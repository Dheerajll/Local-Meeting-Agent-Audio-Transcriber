"""
manual_test.py — run the recorder against your mic/BlackHole input and dump
every chunk it publishes, so you can eyeball chunk boundaries/reasons before
wiring a real transcriber in.
"""

import wave
from pathlib import Path

from lma.audio import AudioRecorder
from lma.publisher import ChunkPublisher
from lma.queues import transcription_queue

OUT_DIR = Path("test_chunks")
OUT_DIR.mkdir(exist_ok=True)

publisher = ChunkPublisher(transcription_queue)
recorder = AudioRecorder(publisher)

print("Recording... speak, pause, speak again. Ctrl+C to stop.")
try:
    recorder.start()  # blocks — runs loop() until you interrupt
except KeyboardInterrupt:
    recorder.stop()

print(f"\n{transcription_queue.qsize()} chunk(s) published:\n")

i = 0
while not transcription_queue.empty():
    chunk = transcription_queue.get()
    i += 1

    # Write pcm_bytes out as a real .wav so you can actually listen to it —
    # the recorder itself never does this, this is purely a test harness.
    wav_path = OUT_DIR / f"chunk_{chunk.chunk_id:03d}.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(chunk.channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(chunk.sample_rate)
        wf.writeframes(chunk.pcm_bytes)

    print(
        f"[{i}] id={chunk.chunk_id} "
        f"reason={chunk.reason.value} forced={chunk.forced} "
        f"start={chunk.start_ms}ms end={chunk.end_ms}ms "
        f"overlap={chunk.overlap_ms}ms "
        f"duration={chunk.end_ms - chunk.start_ms}ms "
        f"-> {wav_path}"
    )