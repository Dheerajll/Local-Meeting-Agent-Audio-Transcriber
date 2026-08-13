import sys

from .setup import run_setup
from .browser.auth import login
from .browser import BrowserManager
from lma.audio.recorder import AudioRecorder
from lma.workers.publisher import ChunkPublisher
from lma.core.queues import transcription_queue
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

        '''
        recorder = AudioRecorder(ChunkPublisher(transcription_queue))

        try:
            recorder.start()

        except KeyboardInterrupt:

            recorder.stop()
        '''

    else:

        print(
            f"Unknown command: {command}"
        )