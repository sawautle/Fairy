#!/usr/bin/env python3
"""
Tests for the Push-to-Talk state machine in controller/voice_listener.py.

Covers:
  - Rising/falling edge detection via is_ptt_active mocking.
  - Buffer accumulation and finalization on PTT release.
  - Short taps (< 300 ms) are discarded.
  - "● rec" indicator printed on rising edge (wrapped in try/except).
  - _transcribe is called exactly once per valid utterance.
  - print failures on the rising edge never break audio processing.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from controller import voice_listener
from controller.voice_listener import (
    MIN_UTTERANCE_SEC,
    SAMPLE_RATE,
    FRAME_SAMPLES,
    VoiceListener,
)


# ---------------------------------------------------------------------------
# Test subclass — overrides _transcribe and _finalize so tests don't need
# to patch module-level functions that may already be resolved in closures.
# ---------------------------------------------------------------------------

class PTTListenerUnderTest(VoiceListener):
    """
    VoiceListener subclass with injectable behaviour for testing.

    Overrides _transcribe and _finalize so tests don't need to patch module-level
    functions that may already be resolved in daemon-thread closures.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._xcribe_mock = MagicMock(return_value="hello")
        self._finalized_pcm: bytes | None = None
        self._finalize_called = False

    def _transcribe(self, pcm_frames: bytes) -> str:
        """Called by _finalize; delegates to the injectable mock."""
        return self._xcribe_mock(pcm_frames)

    def _finalize(self, buffer: bytearray):
        pcm = bytes(buffer)
        min_bytes = int(SAMPLE_RATE * 2 * MIN_UTTERANCE_SEC)
        if len(pcm) < min_bytes:
            return
        self._finalized_pcm = pcm
        self._finalize_called = True
        text = self._transcribe(pcm)
        if text:
            self.on_directed(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_bytes(n_frames: int = 1) -> bytes:
    """Return n_frames of silent 16-bit PCM as bytes."""
    return b"\x00\x00" * FRAME_SAMPLES * n_frames


def _inject_frames(listener: VoiceListener, frames: bytes):
    """Write raw bytes into the listener's audio queue in small chunks."""
    chunk_size = FRAME_SAMPLES * 2  # 960 bytes per 30 ms frame
    for i in range(0, len(frames), chunk_size):
        listener._audio_q.put(frames[i : i + chunk_size])


def _run_process_loop(listener: VoiceListener, ptt_seq: list, audio: bytes,
                      timeout: float = 5.0):
    """
    Run _process_loop in a daemon thread; stop after ptt_seq exhausted.
    All audio is queued BEFORE the thread starts so _process_loop never
    races the main thread on the queue.
    """
    ptt_iter = iter(ptt_seq)

    def ptt_provider():
        try:
            return next(ptt_iter)
        except StopIteration:
            listener._stop_event.set()
            return False

    def run():
        with patch("controller.voice_listener.is_ptt_active", ptt_provider):
            with patch("controller.voice_listener.is_speaking", return_value=False):
                _inject_frames(listener, audio)
                listener._process_loop()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        listener._stop_event.set()
        t.join(timeout=2)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_sounddevice():
    """Stub sounddevice import so VoiceListener can be instantiated."""
    with patch.dict("sys.modules", {"sounddevice": MagicMock()}):
        yield


@pytest.fixture
def listener(mock_sounddevice):
    """A PTTListenerUnderTest with a fresh audio queue."""
    directed = MagicMock()
    return PTTListenerUnderTest(on_directed=directed, on_noteworthy=None)


# ---------------------------------------------------------------------------
# State machine tests
# ---------------------------------------------------------------------------

class TestPTTRisingEdge:
    """Indicator printed on rising edge (press)."""

    def test_indicator_printed_on_rising_edge(self, listener, capsys, monkeypatch):
        """
        When PTT transitions False→True and audio frames are queued,
        the "● rec" indicator must be printed to stdout.

        Explicitly set PTT_QUIET=False for this test so it exercises the
        print path regardless of the production default. The test should
        not depend on whether the runtime default is True or False.

        ptt_seq starts with True so the very first frame is treated as a
        rising edge (no pre-roll frame is discarded).  We use enough frames to
        clear the MIN_UTTERANCE threshold so the loop also reaches the falling
        edge — that way the print+flush path runs in a daemon thread where
        pytest's capsys can capture it.
        """
        monkeypatch.setattr(voice_listener, "PTT_QUIET", False)

        # 12 frames @ 30 ms = 360 ms > MIN_UTTERANCE_SEC=0.3
        ptt_seq = [True] * 12 + [False]
        audio = _frame_bytes(12)
        _run_process_loop(listener, ptt_seq, audio)

        time.sleep(0.05)
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "● rec" in combined, (
            f"Expected '● rec' in captured output, got: {combined!r}"
        )

    def test_indicator_suppressed_when_quiet(self, listener, capsys):
        """
        With PTT_QUIET=True (the production default), a full press+release
        cycle must produce NEITHER the "● rec" indicator NOR the
        "released" release message. Only the transcript line should appear,
        and that line is emitted by the test subclass's _finalize, not by
        _process_loop — so stdout from _process_loop must be empty.
        """
        # PTT_QUIET default is True; no monkeypatch needed.
        ptt_seq = [True] * 12 + [False]
        audio = _frame_bytes(12)
        _run_process_loop(listener, ptt_seq, audio)

        time.sleep(0.05)
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "● rec" not in combined, (
            f"Did not expect '● rec' when PTT_QUIET=True, got: {combined!r}"
        )
        assert "released" not in combined, (
            f"Did not expect 'released' message when PTT_QUIET=True, "
            f"got: {combined!r}"
        )


class TestPTTFallingEdge:
    """Buffer finalized and transcribed on falling edge (release)."""

    def test_press_then_release_sends_audio(self, listener):
        """
        Press PTT → record several frames → release PTT.
        _finalize must be called once with the accumulated PCM,
        and _transcribe must receive those bytes.
        """
        # 12 frames = 360 ms > MIN_UTTERANCE_SEC=0.3
        ptt_seq = [True] * 12 + [False]
        audio = _frame_bytes(12)
        _run_process_loop(listener, ptt_seq, audio)

        assert listener._finalize_called, "_finalize was not called on PTT release"
        assert listener._finalized_pcm is not None, "_finalized_pcm is None"
        expected_bytes = 12 * FRAME_SAMPLES * 2
        assert len(listener._finalized_pcm) == expected_bytes, (
            f"Expected {expected_bytes} bytes in finalization buffer, "
            f"got {len(listener._finalized_pcm)}"
        )
        listener._xcribe_mock.assert_called_once()
        listener.on_directed.assert_called_once_with("hello")


class TestShortTapDiscard:
    """Taps shorter than MIN_UTTERANCE_SEC are silently dropped."""

    def test_tap_under_300ms_discarded(self, listener):
        """
        Press and release in under MIN_UTTERANCE_SEC.
        _transcribe must NOT be called; _finalize must NOT fire.
        """
        # 1 frame = 30 ms < 300 ms threshold.  True→False: rising then falling.
        ptt_seq = [True, False]
        audio = _frame_bytes(1)
        _run_process_loop(listener, ptt_seq, audio)

        listener._xcribe_mock.assert_not_called()
        assert not listener._finalize_called


class TestBufferAccumulation:
    """Multi-frame buffer built up correctly before finalization."""

    def test_long_utterance_accumulates(self, listener):
        """
        Hold PTT for longer than MIN_UTTERANCE_SEC, then release.
        The PCM accumulated in _finalize must span all frames.
        """
        # 11 frames = 330 ms ≥ MIN_UTTERANCE_SEC=0.3 (9600-byte threshold)
        ptt_seq = [True] * 11 + [False]
        audio = _frame_bytes(11)
        _run_process_loop(listener, ptt_seq, audio)

        assert listener._finalize_called
        expected_bytes = 11 * FRAME_SAMPLES * 2
        assert len(listener._finalized_pcm) == expected_bytes, (
            f"Expected {expected_bytes} bytes, got {len(listener._finalized_pcm)}"
        )


class TestEdgeResilience:
    """The rising-edge print is wrapped in try/except — never crashes."""

    def test_indicator_print_failure_does_not_break_processing(self, listener):
        """
        Simulate print() raising an exception.
        Processing must continue and _finalize must still be called.
        """
        # 11 frames = 330 ms ≥ MIN_UTTERANCE_SEC=0.3
        ptt_seq = [True] * 11 + [False]
        audio = _frame_bytes(11)
        ptt_iter = iter(ptt_seq)

        def ptt_provider():
            try:
                return next(ptt_iter)
            except StopIteration:
                listener._stop_event.set()
                return False

        def run():
            with patch("controller.voice_listener.is_ptt_active", ptt_provider):
                with patch("controller.voice_listener.is_speaking", return_value=False):
                    with patch("builtins.print", side_effect=OSError("stdout broken")):
                        _inject_frames(listener, audio)
                        listener._process_loop()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=5.0)
        if t.is_alive():
            listener._stop_event.set()
            t.join(timeout=2)

        # Processing must complete even though print failed
        assert listener._finalize_called, "Processing did not finalize after print failure"
        listener.on_directed.assert_called_once_with("hello")
