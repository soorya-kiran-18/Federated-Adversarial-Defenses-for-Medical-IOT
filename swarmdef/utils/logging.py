"""Console logging with a consistent, demo-friendly format."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False

# ANSI colours for the live demo terminal.
COLORS = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;196m",
    "CRITICAL": "\033[48;5;196m\033[97m",
}
RESET = "\033[0m"
BOLD = "\033[1m"


class _Formatter(logging.Formatter):
    def __init__(self, color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self.color:
            return f"{COLORS.get(record.levelname, '')}{text}{RESET}"
        return text


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Install the root handler once; safe to call from any entry point."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger()
    root.setLevel(level)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(_Formatter(color=sys.stdout.isatty()))
    root.addHandler(stream)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(_Formatter(color=False))
        root.addHandler(fh)

    # Third-party libraries are chatty during federated simulation.
    for noisy in ("flwr", "matplotlib", "opacus", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def banner(text: str, char: str = "=", width: int = 78) -> str:
    """A section header for the step-by-step demo output."""
    pad = max(0, width - len(text) - 2)
    left = pad // 2
    return f"\n{BOLD}{char * left} {text} {char * (pad - left)}{RESET}"
