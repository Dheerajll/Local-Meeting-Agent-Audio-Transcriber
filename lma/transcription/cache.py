from pathlib import Path

from lma.paths import HF_CACHE_DIR

def _find_snapshot(snapshots_dir:Path,model_name:str) -> Path:
    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"No snapshots found for '{model_name}'."
        )

    snapshots = sorted(
        p
        for p in snapshots_dir.iterdir()
        if p.is_dir()
    )

    if not snapshots:
        raise FileNotFoundError(
            f"No snapshot available for '{model_name}'."
        )

    #
    # HF snapshots are immutable.
    # Returning the newest one is sufficient.
    #
    return snapshots[-1]

def get_model_snapshot_path(model_name: str) -> Path:
    """
    Returns the local snapshot directory for a Hugging Face model.

    Expected layout:

        HF_CACHE_DIR/
            models--mlx-community--whisper-large-v3-mlx/
                snapshots/
                    <hash>/

    Raises:
        FileNotFoundError:
            If the model or snapshot does not exist.
    """

    model_dir = (
        HF_CACHE_DIR
        / f"models--{model_name.replace('/', '--')}"
    )

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model '{model_name}' not found.\n"
            f"Expected: {model_dir}"
        )

    snapshots_dir = model_dir / "snapshots"
    """
    Returns:
        Path to the snapshot directory.
    """
    return _find_snapshot(snapshots_dir,model_name)