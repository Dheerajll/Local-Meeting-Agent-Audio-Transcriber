"""
Browser automation manager.

Handles:
- Launching Playwright with a persistent Chrome profile
- Joining meetings (Google Meet for now)
- Detecting whether the bot actually entered the meeting
- Clean shutdown
"""

import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout  # type: ignore

from lma.core.paths import (
    BROWSER_PROFILE_DIR,
    ensure_directories,
)
from lma.core.exceptions import BrowserError


class BrowserManager:

    def __init__(self):
        ensure_directories()
        self.playwright = None
        self.context = None
        self.page = None

    # ================================================================
    # LIFECYCLE
    # ================================================================

    def start(self):
        """Launch Playwright and open a persistent Chrome context."""
        try:
            self.playwright = sync_playwright().start()
        except Exception as exc:
            raise BrowserError(f"Failed to start Playwright: {exc}")

        try:
            self.context = (
                self.playwright
                .chromium
                .launch_persistent_context(
                    user_data_dir=str(BROWSER_PROFILE_DIR),
                    headless=False,
                    channel="chrome",
                    args=[
                        "--use-fake-ui-for-media-stream",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
            )
        except Exception as exc:
            self._cleanup_playwright()
            raise BrowserError(f"Failed to launch browser: {exc}")

        self.page = (
            self.context.pages[0]
            if self.context.pages
            else self.context.new_page()
        )

        return self.page

    def close(self):
        """Close browser and Playwright, tolerating partial failures."""
        if self.context:
            try:
                self.context.close()
            except Exception as exc:
                print(f"⚠️  Error closing context: {exc}")
            self.context = None

        self._cleanup_playwright()

    def _cleanup_playwright(self):
        if self.playwright:
            try:
                self.playwright.stop()
            except Exception as exc:
                print(f"⚠️  Error stopping Playwright: {exc}")
            self.playwright = None

    # ================================================================
    # MEETING JOIN
    # ================================================================

    def join_meeting(self, url: str) -> bool:
        """
        Navigate to the meeting URL, turn off camera/mic,
        click join, and wait until the bot is inside the meeting.

        Returns True if the bot successfully entered the meeting.
        Raises BrowserError on critical failures.
        """
        if self.page is None:
            raise BrowserError("Browser not started. Call start() first.")

        # ----------------------------------------------------------
        # 1. Navigate
        # ----------------------------------------------------------
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except PlaywrightTimeout:
            raise BrowserError(f"Timed out loading meeting URL: {url}")

        # Give the page a moment to render dynamic content.
        self.page.wait_for_timeout(2000)

        # ----------------------------------------------------------
        # 2. Turn off camera and microphone
        # ----------------------------------------------------------
        self._try_click_button("Turn off camera")
        self._try_click_button("Turn off microphone")

        # ----------------------------------------------------------
        # 3. Click the join button
        # ----------------------------------------------------------
        joined = self._click_join_button()
        if not joined:
            raise BrowserError(
                "Could not find any join button on the page."
            )

        # ----------------------------------------------------------
        # 4. Wait to actually enter the meeting room
        # ----------------------------------------------------------
        entered = self._wait_for_meeting_room()
        if entered:
            print("✓ Bot entered the meeting room")
        else:
            print("⚠️  Could not confirm meeting entry")

        return entered

    # ================================================================
    # PRIVATE HELPERS
    # ================================================================

    def _try_click_button(self, name: str, timeout: int = 3000) -> bool:
        """Try to click a button by its accessible name. Fail silently."""
        try:
            self.page.get_by_role("button", name=name).click(timeout=timeout)
            return True
        except Exception:
            return False

    def _click_join_button(self) -> bool:
        """
        Try multiple known join button labels.
        Returns True if any were clicked.
        """
        join_labels = [
            "Join now",
            "Ask to join",
            "Join",
            "Join meeting",
        ]

        for label in join_labels:
            try:
                self.page.get_by_role(
                    "button",
                    name=label,
                ).click(timeout=5000)
                print(f"✓ Clicked '{label}'")
                return True
            except Exception:
                continue

        return False

    def _wait_for_meeting_room(self, timeout_ms: int = 30_000) -> bool:
        """
        After clicking join, wait until the meeting room is visible.

        For Google Meet, the meeting room contains:
        - A "Leave call" button
        - A microphone toggle
        - Chat / participants controls

        We look for the "Leave call" button as the strongest signal
        that we are inside the meeting.

        If the bot was placed in a lobby ("Ask to join"), we wait
        for the host to admit us, then re-check.
        """
        deadline = time.time() + (timeout_ms / 1000)

        while time.time() < deadline:

            # Check if we're in the meeting room.
            if self._is_in_meeting():
                return True

            # Check if we're stuck in a lobby.
            if self._is_in_lobby():
                print("⏳ In lobby, waiting for host to admit...")
                self.page.wait_for_timeout(3000)
                continue

            self.page.wait_for_timeout(1000)

        return False

    def _is_in_meeting(self) -> bool:
        """Check for indicators that we are inside the meeting room."""
        indicators = [
            "Leave call",
            "Turn off microphone",
            "Turn on microphone",
        ]
        for text in indicators:
            try:
                count = self.page.get_by_role(
                    "button", name=text
                ).count()
                if count > 0:
                    return True
            except Exception:
                continue
        return False

    def _is_in_lobby(self) -> bool:
        """Check if we are in a waiting / lobby screen."""
        lobby_indicators = [
            "Waiting for host",
            "Someone will let you in",
            "You're in the lobby",
        ]
        for text in lobby_indicators:
            try:
                count = self.page.get_by_text(text, exact=False).count()
                if count > 0:
                    return True
            except Exception:
                continue
        return False