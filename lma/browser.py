from playwright.sync_api import sync_playwright

from .paths import (
    BROWSER_PROFILE_DIR,
    ensure_directories
)

class BrowserManager:

    def __init__(self):
        ensure_directories()
        self.playwright = None
        self.context = None
        self.page = None

    def start(self):

        self.playwright = (
            sync_playwright()
            .start()
        )

        self.context = (
            self.playwright
            .chromium
            .launch_persistent_context(
                user_data_dir=str(
                    BROWSER_PROFILE_DIR
                ),

                headless=False,
                channel="chrome",
                args=[
                    "--use-fake-ui-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                    "--disable-blink-features=AutomationControlled"
                ]
            )
        )
        self.page = (
            self.context.pages[0]
            if self.context.pages
            else self.context.new_page()
        )

        return self.page

    def close(self):

        if self.context:
            self.context.close()

        if self.playwright:
            self.playwright.stop()


    def join_meeting(self, url):

        self.page.goto(url)


        self.page.wait_for_load_state("networkidle")


        # camera off
        try:
            self.page.get_by_role(
                "button",
                name="Turn off camera"
            ).click(timeout=3000)

        except:
            pass


        # microphone off
        try:
            self.page.get_by_role(
                "button",
                name="Turn off microphone"
            ).click(timeout=3000)

        except:
            pass



        # join
        try:

            self.page.get_by_role(
                "button",
                name="Join now"
            ).click(
                timeout=5000
            )


        except:

            try:

                self.page.get_by_role(
                    "button",
                    name="Ask to join"
                ).click(
                    timeout=5000
                )

            except:

                print(
                    "Join button not found"
                )