import mlx.core as mx

from mlx_whisper.load_models import load_model

from lma.transcription.cache import get_model_path


def create_whisper_model(model_name: str):
    """
    Load a Whisper model from the local Hugging Face snapshot.

    The model is loaded into memory once by the caller
    and can then be reused for multiple transcription requests.
    """

    model_path = get_model_path(model_name)

    print(f"Loading Whisper model from: {model_path}")

    model = load_model(
        str(model_path),
        dtype=mx.float32,
    )

    print("✓ Whisper model loaded")

    return model