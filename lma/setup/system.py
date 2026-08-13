import subprocess
import shutil


def run_command(command):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0

    except Exception:
        return False

# -----------------------------
# Homebrew
# -----------------------------

def check_homebrew():

    brew = shutil.which("brew")
    if brew:
        print("✓ Homebrew found")
        return True

    print("✗ Homebrew not found")
    return False

# -----------------------------
# FFmpeg
# -----------------------------

def check_ffmpeg():

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(
            f"✓ FFmpeg found: {ffmpeg}"
        )
        return True
    print("✗ FFmpeg missing")

    return False



def install_ffmpeg():

    print("Installing FFmpeg...")
    success = run_command(
        [
            "brew",
            "install",
            "ffmpeg"
        ]
    )
    if success:
        print("✓ FFmpeg installed")
    else:
        print("✗ FFmpeg installation failed")

    return success
# -----------------------------
# BlackHole
# -----------------------------

def check_blackhole():

    result = subprocess.run(
        [
            "brew",
            "list",
            "blackhole-2ch"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:

        print("✓ BlackHole 2ch found")
        return True
    
    print("✗ BlackHole missing")

    return False



def install_blackhole():

    print("Installing BlackHole 2ch...")

    success = run_command(
        [
            "brew",
            "install",
            "--cask",
            "blackhole-2ch"
        ]
    )
    if success:
        print(
            "✓ BlackHole installed"
        )
        print(
            "⚠ Restart may be required"
        )
    else:
        print(
            "✗ BlackHole installation failed"
        )
    return success

# -----------------------------
# switchaudio-osx
# -----------------------------

def check_switch_audio_source():

    switch_audio = shutil.which(
        "SwitchAudioSource"
    )

    if switch_audio:

        print(
            f"✓ switchaudio-osx found: {switch_audio}"
        )

        return True

    print(
        "✗ switchaudio-osx missing"
    )

    return False



def install_switch_audio_source():

    print(
        "Installing switchaudio-osx..."
    )

    success = run_command(
        [
            "brew",
            "install",
            "switchaudio-osx"
        ]
    )


    if success:

        print(
            "✓ switchaudio-osx installed"
        )

    else:

        print(
            "✗ switchaudio-osx installation failed"
        )


    return success