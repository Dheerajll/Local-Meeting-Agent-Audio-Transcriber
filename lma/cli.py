import sys

from .setup import run_setup
from .auth import login
from .browser import BrowserManager
from .audio import AudioRecorder

def main():

    if len(sys.argv) < 2:

        print(
            """
Usage:
    lma setup
    lma login
    lma join <meeting_url>
            """
        )

        return


    command = sys.argv[1]


    if command == "setup":

        run_setup()


    elif command == "login":

        login()


    elif command == "join":

        if len(sys.argv) < 3:

            print(
                "Usage: lma join <meeting_url>"
            )

            return


        meeting_url = sys.argv[2]


        browser = BrowserManager()

        try:

            browser.start()

            browser.join_meeting(
                meeting_url
            )


            input(
                "Press ENTER to close..."
            )


        finally:

            browser.close()
    
    elif command == "record":


        recorder = AudioRecorder()

        try:
            recorder.start()

        except KeyboardInterrupt:

            recorder.stop()


    else:

        print(
            f"Unknown command: {command}"
        )