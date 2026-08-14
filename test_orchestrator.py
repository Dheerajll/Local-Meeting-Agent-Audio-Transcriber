from lma.orchestration.orchestrator import MeetingOrchestrator

if __name__ == "__main__":
    # Replace with a real meeting link you can test with
    url = "https://meet.google.com/xxx-xxxx-xxx" 
    
    orchestrator = MeetingOrchestrator(
        meeting_url=url,
        session_id="manual_test"
    )
    
    orchestrator.run()