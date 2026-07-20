import subprocess


class AudioDeviceManager:


    def __init__(self):

        self.previous_device = None


    def get_current_device(self):

        result = subprocess.run(
            [
                "SwitchAudioSource",
                "-c"
            ],
            capture_output=True,
            text=True,
            check=True
        )

        return result.stdout.strip()



    def switch_to(self, device_name):

        subprocess.run(
            [
                "SwitchAudioSource",
                "-s",
                device_name
            ],
            check=True
        )


    def start_recording_mode(
        self,
        recording_device="Local meeting agent output"
    ):

        self.previous_device = (
            self.get_current_device()
        )


        print(
            f"Previous device: {self.previous_device}"
        )


        print(
            f"Switching to {recording_device}"
        )


        self.switch_to(
            recording_device
        )



    def stop_recording_mode(self):

        if self.previous_device:

            print(
                f"Restoring {self.previous_device}"
            )

            self.switch_to(
                self.previous_device
            )
