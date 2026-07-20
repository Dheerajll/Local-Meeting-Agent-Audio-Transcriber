import os

from huggingface_hub import snapshot_download

from .paths import HF_CACHE_DIR


WHISPER_REPO = (
    "mlx-community/"
    "whisper-large-v3-mlx"
)


def configure_hf_cache():

    """
    Redirect HuggingFace cache
    into Local Meeting Agent storage.
    """

    os.environ["HF_HOME"] = str(
        HF_CACHE_DIR
    )


def download_whisper_model():

    configure_hf_cache()

    print(
        "\nDownloading Whisper large-v3 MLX model..."
    )

    print(
        f"Cache location:"
        f"\n{HF_CACHE_DIR}"
    )


    model_path = snapshot_download(
        repo_id=WHISPER_REPO,
        cache_dir=str(HF_CACHE_DIR)
    )


    print(
        "\n✓ Whisper model ready"
    )

    print(
        model_path
    )

    return model_path



def check_whisper_model():

    configure_hf_cache()

    if not HF_CACHE_DIR.exists():

        return False


    # Search for downloaded model

    for path in HF_CACHE_DIR.rglob("*"):

        if "whisper-large-v3-mlx" in str(path):

            return True


    return False



def setup_whisper():

    if check_whisper_model():

        print(
            "✓ Whisper large-v3 already exists"
        )

        return


    download_whisper_model()