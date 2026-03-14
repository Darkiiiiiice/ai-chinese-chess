"""Logging utilities with worker identification"""

import os
import threading
from typing import Optional


def get_worker_id() -> str:
    """Get current worker identifier (process and thread)"""
    pid = os.getpid()
    tid = threading.current_thread().ident
    return f"P{pid}-T{tid}"


def get_short_worker_id() -> str:
    """Get short worker identifier (just PID)"""
    return f"P{os.getpid()}"


def log(message: str, worker_id: Optional[str] = None) -> str:
    """
    Format log message with worker identifier.

    Args:
        message: Log message
        worker_id: Optional custom worker ID (e.g., "Worker-1")

    Returns:
        Formatted message with worker prefix
    """
    if worker_id:
        prefix = f"[{worker_id}]"
    else:
        prefix = f"[{get_short_worker_id()}]"

    return f"{prefix} {message}"


def wprint(message: str, worker_id: Optional[str] = None):
    """Print with worker identifier prefix"""
    print(log(message, worker_id))
