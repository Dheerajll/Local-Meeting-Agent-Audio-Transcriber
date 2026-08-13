import subprocess


REQUIRED_DEVICES = [
    "BlackHole 2ch",
    "Local meeting agent output"
]


def get_audio_devices():

    result = subprocess.run(
        [
            "SwitchAudioSource",
            "-a"
        ],
        capture_output=True,
        text=True,
        check=True
    )

    return [
        x.strip()
        for x in result.stdout.splitlines()
    ]



def check_audio_setup():

    devices = get_audio_devices()

    print("\nChecking audio devices...")

    success = True

    for device in REQUIRED_DEVICES:

        if device in devices:

            print(
                f"✓ {device}"
            )

        else:

            print(
                f"✗ Missing: {device}"
            )

            success = False


    return success