import sys


def main():
    if len(sys.argv) < 2:
        _print_usage()
        return

    command = sys.argv[1]

    if command == "setup":
        from lma.setup.runner import run_setup
        run_setup()

    elif command == "login":
        from lma.browser.auth import login
        login()

    elif command == "config":
        _handle_config()

    elif command == "join":
        _handle_join()
    
    elif command == "daemon":
        from lma.daemon import run_daemon
        run_daemon()

    else:
        print(f"Unknown command: {command}")
        _print_usage()


def _print_usage():
    print("""
Usage:
  lma setup                          Install dependencies
  lma login                          Login to Google (browser)
  lma config set-token <token>       Set your LMA authentication token
  lma config set-backend <url>       Set the backend server URL
  lma config show                    Show current configuration
  lma config verify                  Verify token with backend
  lma join <meeting_url>             Join and record a meeting
  lma daemon                         Start daemon (listens for backend commands)
""")

def _handle_config():
    if len(sys.argv) < 3:
        print("Usage: lma config <set-token|set-backend|show|verify>")
        return

    subcommand = sys.argv[2]

    from lma.core.config import (
        load_config,
        set_token,
        set_backend_url,
        get_backend_url,
    )
    from lma.core.exceptions import ConfigError

    if subcommand == "set-token":
        if len(sys.argv) < 4:
            print("Usage: lma config set-token <token>")
            return
        token = sys.argv[3]
        try:
            set_token(token)
            # Mask the token for display
            masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
            print(f"✓ Token saved: {masked}")
        except ConfigError as exc:
            print(f"✗ {exc}")

    elif subcommand == "set-backend":
        if len(sys.argv) < 4:
            print("Usage: lma config set-backend <url>")
            return
        url = sys.argv[3]
        try:
            set_backend_url(url)
            print(f"✓ Backend URL saved: {url}")
        except ConfigError as exc:
            print(f"✗ {exc}")

    elif subcommand == "show":
        config = load_config()
        token_display = "Not set"
        if config.lma_token:
            t = config.lma_token
            token_display = t[:8] + "..." + t[-4:] if len(t) > 12 else "***"

        print(f"""
┌─────────────────────────────────────────┐
│  LMA Configuration                      │
├─────────────────────────────────────────┤
│  Backend URL:  {config.backend_url:<25}│
│  LMA Token:    {token_display:<25}│
│  Device Name:  {config.device_name:<25}│
│  Configured:   {'Yes' if config.lma_token else 'No':<25}│
└─────────────────────────────────────────┘
        """)

    elif subcommand == "verify":
        _verify_token()

    else:
        print(f"Unknown config command: {subcommand}")


def _verify_token():
    """Ping the backend to verify the token is valid."""
    import httpx #type:ignore
    from lma.core.config import load_config
    from lma.core.exceptions import ConfigError

    config = load_config()

    if not config.lma_token:
        print("✗ No token configured. Run: lma config set-token <token>")
        return

    url = f"{config.backend_url}/api/v1/auth/lma/verify"
    headers = {"Authorization": f"Bearer {config.lma_token}"}

    print(f"Verifying token with {config.backend_url}...")

    try:
        response = httpx.get(url, headers=headers, timeout=10.0)

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Token valid!")
            print(f"  User: {data.get('user_email', 'unknown')}")
            print(f"  Device: {data.get('device_name', 'unknown')}")
        elif response.status_code == 401:
            print("✗ Token rejected (invalid or revoked)")
        elif response.status_code == 404:
            print("✗ Backend endpoint not found. Is the backend running?")
        else:
            print(f"✗ Unexpected response: {response.status_code}")

    except httpx.ConnectError:
        print(f"✗ Cannot connect to {config.backend_url}")
        print("  Is the backend server running?")
    except Exception as exc:
        print(f"✗ Verification failed: {exc}")


def _handle_join():
    if len(sys.argv) < 3:
        print("Usage: lma join <meeting_url> [--meeting-id=123] [--lang=en]")
        return

    meeting_url = sys.argv[2]

    language = None
    backend_meeting_id = None  # <-- NEW
    
    for arg in sys.argv[3:]:
        if arg.startswith("--lang="):
            language = arg.split("=", 1)[1]
        elif arg.startswith("--meeting-id="):
            backend_meeting_id = int(arg.split("=", 1)[1]) # <-- NEW

    from lma.orchestration.orchestrator import MeetingOrchestrator

    orchestrator = MeetingOrchestrator(
        meeting_url=meeting_url,
        backend_meeting_id = backend_meeting_id,  # <-- NEW
        language=language,
    )

    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
        orchestrator.stop()


if __name__ == "__main__":
    main()