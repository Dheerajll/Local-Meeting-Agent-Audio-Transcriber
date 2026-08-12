import inspect
import mlx_whisper

print(inspect.signature(mlx_whisper.transcribe))
print()
print(inspect.getsource(mlx_whisper.transcribe))