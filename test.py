from lma.transcription.factory import create_whisper_model


MODEL_NAME = "mlx-community/whisper-large-v3-mlx"


model = create_whisper_model(MODEL_NAME)

print()
print("Model type:")
print(type(model))

print()
print("Model loaded successfully")