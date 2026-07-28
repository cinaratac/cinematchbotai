import asyncio
import os
import tempfile
import uuid
import wave

from database import save_voice_recording
from evaluation_service import evaluate_voice_session


class VoiceRecordingSession:
    """Bir WebSocket görüşmesinin kullanıcı ve agent PCM kanallarını yönetir."""

    def __init__(self, session_id, user_id, username, input_sample_rate):
        self.recording_id = uuid.uuid4().hex
        self.session_id = session_id
        self.user_id = user_id
        self.username = username
        self.input_sample_rate = input_sample_rate
        self.paths = {}
        self.writers = {}
        self.byte_counts = {"user": 0, "agent": 0}

        for track, sample_rate in (("user", input_sample_rate), ("agent", 24000)):
            temp_file = tempfile.NamedTemporaryFile(
                prefix=f"cinematch-{track}-",
                suffix=".wav",
                delete=False,
            )
            temp_file.close()
            writer = wave.open(temp_file.name, "wb")
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            self.paths[track] = temp_file.name
            self.writers[track] = writer

    @classmethod
    def create_if_enabled(
        cls,
        session_id,
        user_id,
        username,
        input_sample_rate,
    ):
        enabled = (
            os.environ.get("VOICE_RECORDING_ENABLED", "true").lower()
            not in {"0", "false", "no"}
        )
        bucket_configured = bool(
            os.environ.get("FIREBASE_STORAGE_BUCKET", "").strip()
        )
        if enabled and not bucket_configured:
            print(
                "VOICE KAYIT UYARISI: FIREBASE_STORAGE_BUCKET tanımlı değil; "
                "bu görüşme kaydedilmeyecek."
            )
            return None
        if not enabled:
            return None
        print("Voice kaydı başlatıldı; Firebase Storage bucket hazır.")
        return cls(session_id, user_id, username, input_sample_rate)

    def write(self, track, pcm_bytes):
        writer = self.writers.get(track)
        if writer and pcm_bytes:
            writer.writeframesraw(pcm_bytes)
            self.byte_counts[track] += len(pcm_bytes)

    async def finalize(self):
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception:
                pass

        try:
            if self.byte_counts["user"] <= 0:
                return
            user_duration_ms = round(
                self.byte_counts["user"]
                / (self.input_sample_rate * 2)
                * 1000
            )
            agent_duration_ms = round(
                self.byte_counts["agent"] / (24000 * 2) * 1000
            )
            await asyncio.to_thread(
                save_voice_recording,
                self.session_id,
                self.user_id,
                self.username,
                self.recording_id,
                self.paths["user"],
                self.paths["agent"],
                user_duration_ms,
                agent_duration_ms,
            )
            print(
                "Voice kaydı Firebase Storage'a yüklendi:",
                self.recording_id,
            )
            if (
                os.environ.get(
                    "VOICE_AI_EVALUATION_ENABLED", "true"
                ).lower()
                not in {"0", "false", "no"}
            ):
                await evaluate_voice_session(
                    self.session_id,
                    self.recording_id,
                )
        except Exception as exc:
            print("Voice kayıt yükleme hatası:", repr(exc))
        finally:
            for path in self.paths.values():
                try:
                    os.unlink(path)
                except (FileNotFoundError, OSError):
                    pass
