import time
from dataclasses import dataclass
from enum import Enum


class BargeInPhase(str, Enum):
    IDLE = "idle"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    RESUMING = "resuming"


@dataclass
class BargeInState:
    """Tek bağlantının test edilebilir barge-in state machine'i."""

    phase: BargeInPhase = BargeInPhase.IDLE
    drop_interrupted_audio: bool = False
    interrupted_user_committed: bool = False
    client_interrupt_started: float | None = None
    last_interrupt_at: float | None = None
    interrupt_count: int = 0
    total_interrupts_this_session: int = 0
    last_interrupt_latency_ms: int | None = None

    def record_interrupt(self, source, now=None, debounce_seconds=0.3):
        now = time.perf_counter() if now is None else now
        if (
            self.last_interrupt_at is not None
            and now - self.last_interrupt_at < debounce_seconds
        ):
            return False
        self.phase = BargeInPhase.INTERRUPTED
        self.drop_interrupted_audio = True
        self.interrupted_user_committed = False
        self.client_interrupt_started = now
        self.last_interrupt_at = now
        self.interrupt_count += 1
        self.total_interrupts_this_session += 1
        print(
            "Barge-in state:",
            f"phase={self.phase.value}",
            f"source={source}",
            f"count={self.total_interrupts_this_session}",
        )
        return True

    def record_user_committed(self):
        self.interrupted_user_committed = True
        self.phase = BargeInPhase.RESUMING

    def record_resume(self, now=None):
        now = time.perf_counter() if now is None else now
        latency_ms = None
        if self.client_interrupt_started is not None:
            latency_ms = round(
                max(0, now - self.client_interrupt_started) * 1000
            )
        self.last_interrupt_latency_ms = latency_ms
        self.client_interrupt_started = None
        self.drop_interrupted_audio = False
        self.interrupted_user_committed = False
        self.phase = BargeInPhase.SPEAKING
        print(
            "Barge-in state:",
            f"phase={self.phase.value}",
            f"latency_ms={latency_ms}",
        )
        return latency_ms

    def record_agent_done(self):
        if self.phase == BargeInPhase.SPEAKING:
            self.phase = BargeInPhase.IDLE
