import inspect

from pyannote.audio import Pipeline


DIARIZATION_MODEL = (
    "pyannote/speaker-diarization-community-1"
)

pipeline = Pipeline.from_pretrained(
    DIARIZATION_MODEL
)

print("=" * 70)
print("RELEVANT PIPELINE METHODS")
print("=" * 70)

for name in dir(pipeline):

    if any(
        keyword in name.lower()
        for keyword in (
            "segment",
            "binar",
            "embedding",
            "diar",
        )
    ):

        print(name)


print("\n" + "=" * 70)
print("SEGMENTATION METHOD SIGNATURES")
print("=" * 70)

for name in dir(pipeline):

    if (
        "segment" in name.lower()
        or "binar" in name.lower()
    ):

        obj = getattr(
            pipeline,
            name,
            None,
        )

        if callable(obj):

            try:
                print(
                    f"\n{name}"
                )
                print(
                    inspect.signature(obj)
                )

            except Exception:
                pass