import json
import shutil
from datetime import datetime
from pathlib import Path
from .paths import (
    create_session_dir
)



class StorageManager:


    def __init__(self):

        self.session_dir = (
            create_session_dir()
        )

        self.chunks_dir = (
            self.session_dir /
            "chunks"
        )


        self.chunks_dir.mkdir(
            exist_ok=True
        )


        self.metadata_file = (
            self.session_dir /
            "chunks.json"
        )


        self.metadata = []



    def save_chunk(
        self,
        wav_path,
        reason,
        forced,
        duration
    ):


        chunk_id = (
            len(self.metadata)
            + 1
        )


        new_name = (
            f"chunk_{chunk_id:03d}.wav"
        )


        destination = (
            self.chunks_dir /
            new_name
        )


        shutil.move(
            wav_path,
            destination
        )


        record = {

            "chunk_id": chunk_id,

            "file": str(destination),

            "created_at":
                datetime.now().isoformat(),

            "reason": reason,

            "forced": forced,

            "duration": duration,

            "status": "recorded"

        }


        self.metadata.append(
            record
        )


        self._save_metadata()



        return record



    def mark_transcribed(
        self,
        chunk_id,
        transcript
    ):


        for item in self.metadata:

            if item["chunk_id"] == chunk_id:

                item["status"] = (
                    "transcribed"
                )

                item["transcript"] = (
                    transcript
                )


        self._save_metadata()



    def mark_uploaded(
        self,
        chunk_id
    ):


        for item in self.metadata:

            if item["chunk_id"] == chunk_id:

                item["status"] = (
                    "uploaded"
                )


        self._save_metadata()



    def delete_audio(
        self,
        chunk_id
    ):


        for item in self.metadata:

            if item["chunk_id"] == chunk_id:


                path = item["file"]


                try:

                    Path(path).unlink()

                except FileNotFoundError:

                    pass


        self._save_metadata()



    def _save_metadata(self):

        with open(
            self.metadata_file,
            "w"
        ) as f:

            json.dump(
                self.metadata,
                f,
                indent=4
            )