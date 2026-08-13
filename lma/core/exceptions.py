class MeetingAgentError(Exception):
    """Base exception."""


class AudioDeviceError(MeetingAgentError):
    pass


class FFmpegError(MeetingAgentError):
    pass


class RecorderError(MeetingAgentError):
    pass


class TranscriptionError(MeetingAgentError):
    pass


class BackendError(MeetingAgentError):
    pass