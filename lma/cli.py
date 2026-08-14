import sys


def main():
    if len(sys.argv) < 2:
        print("""
Usage:
  lma setup
  lma login
  lma join <meeting_url> [--lang=en] [--session=abc]
        """)
        return

    command = sys.argv[1]

    if command == "setup":
        from .setup.runner import run_setup
        run_setup()

    elif command == "login":
        from .browser.auth import login
        login()

    elif command == "join":
        if len(sys.argv) < 3:
            print("Usage: lma join <meeting_url> [--lang=en]")
            return

        meeting_url = sys.argv[2]

        # Parse optional flags
        language = None
        session_id = None
        for arg in sys.argv[3:]:
            if arg.startswith("--lang="):
                language = arg.split("=", 1)[1]
            elif arg.startswith("--session="):
                session_id = arg.split("=", 1)[1]

        from .orchestration.orchestrator import MeetingOrchestrator

        orchestrator = MeetingOrchestrator(
            meeting_url=meeting_url,
            session_id=session_id,
            language=language,
        )

        try:
            orchestrator.run()
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted.")
            orchestrator.stop()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()