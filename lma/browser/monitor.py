"""
Monitors the Playwright page to detect when a meeting has ended.
Runs in a background thread.
"""
import threading
import time
from dataclasses import dataclass
from enum import Enum
from playwright.sync_api import Page # type: ignore

class MeetingEndReason(Enum):
    LEFT_MEETING = "left_meeting"
    URL_CHANGED = "url_changed"
    PAGE_CLOSED = "page_closed"
    MEETING_ENDED = "meeting_ended"
    UNKNOWN = "unknown"

@dataclass
class MonitorResult:
    ended: bool = False
    reason: MeetingEndReason = MeetingEndReason.UNKNOWN
    detail: str = ""

class MeetingMonitor:
    def __init__(
        self,
        page: Page,
        original_url: str,
        poll_interval: float = 5.0,
        grace_period: float = 15.0,
    ):
        self.page = page
        self.original_url = original_url
        self.poll_interval = poll_interval
        self.grace_period = grace_period

        self.meeting_ended = threading.Event()
        self.result = MonitorResult()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("MeetingMonitor is already running.")

        self._stop_event.clear()
        self.meeting_ended.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="meeting-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _run(self) -> None:
        # Grace period: let the page load and transition after joining
        time.sleep(self.grace_period)

        while not self._stop_event.is_set():
            try:
                result = self._check_page()
                if result.ended:
                    self.result = result
                    self.meeting_ended.set()
                    print(f"\n🔴 Meeting ended detected: {result.reason.value} — {result.detail}")
                    return
            except Exception as exc:
                self.result = MonitorResult(
                    ended=True,
                    reason=MeetingEndReason.PAGE_CLOSED,
                    detail=str(exc),
                )
                self.meeting_ended.set()
                print(f"\n🔴 Monitor error (treating as end): {exc}")
                return

            self._stop_event.wait(self.poll_interval)

    def _check_page(self) -> MonitorResult:
        try:
            current_url = self.page.url
        except Exception:
            return MonitorResult(True, MeetingEndReason.PAGE_CLOSED, "Page inaccessible")

        if self._url_changed(current_url):
            return MonitorResult(True, MeetingEndReason.URL_CHANGED, f"URL: {current_url}")

        if self._check_google_meet_left(current_url):
            return MonitorResult(True, MeetingEndReason.LEFT_MEETING, "'You left' screen")

        if self._check_rejoin_button():
            return MonitorResult(True, MeetingEndReason.LEFT_MEETING, "'Rejoin' button")

        if self._check_meeting_ended_indicators():
            return MonitorResult(True, MeetingEndReason.MEETING_ENDED, "Ended indicator")

        return MonitorResult(False)

    def _url_changed(self, current_url: str) -> bool:
        from urllib.parse import urlparse
        original = urlparse(self.original_url)
        current = urlparse(current_url)

        if original.hostname != current.hostname:
            return True
        if original.path.strip("/") and not current.path.strip("/"):
            return True
        return False

    def _check_google_meet_left(self, current_url: str) -> bool:
        if "meet.google.com" not in current_url: return False
        try:
            if self.page.get_by_text("You left the meeting", exact=False).count() > 0:
                return True
        except Exception: pass
        return False

    def _check_rejoin_button(self) -> bool:
        try:
            if self.page.get_by_role("button", name="Rejoin").count() > 0:
                return True
        except Exception: pass
        return False

    def _check_meeting_ended_indicators(self) -> bool:
        indicators = ["This meeting has ended", "The meeting has ended", "Meeting ended", "You have been removed"]
        for text in indicators:
            try:
                if self.page.get_by_text(text, exact=False).count() > 0:
                    return True
            except Exception: continue
        return False