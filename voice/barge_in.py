from dataclasses import dataclass


@dataclass
class BargeInState:
    """Bir streaming bağlantısındaki kesme kapısının değişkenlerini toplar."""

    drop_interrupted_audio: bool = False
    interrupted_user_committed: bool = False
    client_interrupt_started: float | None = None
