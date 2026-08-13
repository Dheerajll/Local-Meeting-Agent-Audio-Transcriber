from pathlib import Path
import shutil

from lma.core.schemas import SessionPaths


class SessionManager:

    def __init__(self, paths: SessionPaths):
        self.paths = paths

    def cleanup(self) -> None:
        if self.paths.root.exists():
            shutil.rmtree(self.paths.root)