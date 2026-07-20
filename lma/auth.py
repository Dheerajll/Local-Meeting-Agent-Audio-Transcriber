from playwright.sync_api import sync_playwright

from .paths import (
    BROWSER_PROFILE_DIR,
    ensure_directories
)



def login():

    ensure_directories()


    print(
        "Opening browser for Google login..."
    )


    with sync_playwright() as p:

        context = p.chromium.launch_persistent_context(
            user_data_dir=str(
                BROWSER_PROFILE_DIR
            ),

            headless=False,

            channel="chrome",

            args=[
                "--use-fake-ui-for-media-stream",
                "--disable-blink-features=AutomationControlled"
            ]
        )


        page = (
            context.pages[0]
            if context.pages
            else context.new_page()
        )


        page.goto(
            "https://accounts.google.com"
        )


        print(
            """
================================================
Login required.

1. Login with your Google account
2. Complete any verification
3. Keep browser open until login finishes
4. Press ENTER here
================================================
"""
        )


        input()


        context.close()


    print(
        "✓ Browser session saved"
    )