"""Единый формат вывода всех стадий конвейера."""

from __future__ import annotations

import sys
import time

_TTY = sys.stdout.isatty()


def _paint(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def step(message: str) -> None:
    print(_paint("1;36", f"→ {message}"), flush=True)


def ok(message: str) -> None:
    print(_paint("32", f"✓ {message}"), flush=True)


def warn(message: str) -> None:
    print(_paint("33", f"! {message}"), file=sys.stderr, flush=True)


def err(message: str) -> None:
    print(_paint("31", f"✗ {message}"), file=sys.stderr, flush=True)


def info(message: str = "") -> None:
    print(message, flush=True)


def kv(key: str, value: object, width: int = 24) -> None:
    print(f"  {key:.<{width}} {value}", flush=True)


def rule(title: str = "") -> None:
    line = "─" * max(4, 60 - len(title))
    print(_paint("2", f"{title} {line}" if title else line), flush=True)


class Progress:
    """Счётчик с частотой обновления не чаще раза в секунду."""

    def __init__(self, title: str, total: int | None = None) -> None:
        self.title = title
        self.total = total
        self.done = 0
        self._started = time.monotonic()
        self._last = 0.0

    def advance(self, count: int = 1, suffix: str = "") -> None:
        self.done += count
        now = time.monotonic()
        if now - self._last < 1.0 and self.done != self.total:
            return
        self._last = now
        elapsed = now - self._started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        total = f"/{self.total}" if self.total else ""
        line = f"  {self.title}: {self.done}{total}  {rate:.1f}/с  {suffix}"
        if _TTY:
            print(f"\r\033[K{line}", end="", flush=True)
        else:
            print(line, flush=True)

    def close(self, suffix: str = "") -> None:
        if _TTY:
            print("\r\033[K", end="")
        elapsed = time.monotonic() - self._started
        print(f"  {self.title}: готово {self.done} за {elapsed:.1f}с {suffix}", flush=True)
