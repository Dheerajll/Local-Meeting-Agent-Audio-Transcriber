from .paths import (
    CACHE_DIR,
    MODEL_DIR,
    LOG_DIR,
    TEMP_DIR
)

from .system import (
    check_homebrew,
    check_ffmpeg,
    install_ffmpeg,
    check_blackhole,
    install_blackhole,
    check_switch_audio_source,
    install_switch_audio_source
)

from .models import setup_whisper
from .audio_setup import check_audio_setup

def create_directories():

    directories = [
        CACHE_DIR,
        MODEL_DIR,
        LOG_DIR,
        TEMP_DIR
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        print(f"✓ {directory}")



def setup_system():

    print("\nChecking system dependencies...\n")

    if not check_homebrew():

        raise RuntimeError(
            "Homebrew required"
        )

    if not check_ffmpeg():
        install_ffmpeg()

    if not check_blackhole():
        install_blackhole()

    if not check_switch_audio_source():
        install_switch_audio_source()
    


def run_setup():

    print("=" * 50)
    print(" Local Meeting Agent Setup")
    print("=" * 50)


    setup_system()


    print(
        "\nCreating runtime directories...\n"
    )

    create_directories()


    print(
        "\nPreparing Whisper model...\n"
    )

    setup_whisper()


    print(
        "\nSetup complete."
    )

    print("Running Local Meeting Agent setup\n")


    audio_ok = check_audio_setup()


    if not audio_ok:

        print(
            """
            Audio setup incomplete.

            Required:
            - BlackHole 2ch
            - Local meeting agent output

            Create them in Audio MIDI Setup.
        """
        )

        return


    print(
        "\n✓ Audio system ready"
    )