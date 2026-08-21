"""
Meeting execution for the daemon.

Runs the MeetingOrchestrator in a blocking fashion.
The orchestrator handles its own lifecycle:
    join meeting → record → transcribe → send chunks → cleanup
"""

from lma.orchestration.orchestrator import MeetingOrchestrator


def run_meeting(
    meeting_id: int,
    meeting_url: str,
    language: str,
) -> None:
    """
    Run the MeetingOrchestrator for a meeting.

    This is a BLOCKING call — it runs until the meeting ends.
    The orchestrator handles its own cleanup on completion
    or interruption (KeyboardInterrupt, meeting end detection).

    Args:
        meeting_id: The backend meeting ID
        meeting_url: The meeting URL to join
        language: The meeting language for transcription
    """
    print(f"\n🎯 Starting meeting {meeting_id}...")
    print(f"   URL: {meeting_url}")
    print(f"   Language: {language}")

    orchestrator = MeetingOrchestrator(
        meeting_url=meeting_url,
        backend_meeting_id=meeting_id,
        language=language,
    )

    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print(f"\n⚠️  Meeting {meeting_id} interrupted by user.")
        orchestrator.stop()
    except Exception as e:
        print(f"❌ Meeting {meeting_id} error: {e}")

    print(f"✅ Meeting {meeting_id} finished.")