#!/usr/bin/env python3
"""
Tests for Whisper STT boot diagnostics and graceful degradation.

Covers:
  1. Model path resolution returns the cached path without hitting the network.
  2. STT-unavailable state is handled gracefully — chat works, PTT returns
     empty, no crash.
  3. Real-path load time: the configured model loads from cache in < 30s.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

# Make the project importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from fairy.py — requires the project's sys.path to be set
from fairy import _resolve_whisper_cache, _is_whisper_running
from controller import voice_listener
from controller.voice_listener import (
    set_stt_ready_getter,
    _is_stt_ready,
    VoiceListener,
)


# ---------------------------------------------------------------------------
# Test 1 — Model path resolution uses the cache and never triggers a download
# ---------------------------------------------------------------------------

class TestWhisperCacheResolution:
    """
    _resolve_whisper_cache must return the correct status and path using
    only local filesystem queries — no network calls are made.

    The network is already blocked by conftest.py's autouse fixture, so any
    HTTP attempt would raise an exception (which would fail the test).
    """

    def test_resolve_returns_cached_when_model_bin_exists(self, monkeypatch):
        """
        When model.bin is present in the snapshot dir, status must be 'cached'
        and snapshot_dir must point to the right place.
        """
        # Pin HF_HOME to the real location so the test uses real cached files
        monkeypatch.setenv("HF_HOME", r"E:\AI\huggingface")
        result = _resolve_whisper_cache("base.en")

        assert result["status"] == "cached", (
            f"Expected 'cached' but got {result['status']!r}. "
            f"HF_HOME={result['hf_home']!r}, "
            f"snapshot_dir={result['snapshot_dir']!r}"
        )
        assert result["model_size"] == "base.en"
        assert result["snapshot_dir"] is not None
        assert os.path.isfile(os.path.join(result["snapshot_dir"], "model.bin"))

    def test_resolve_does_not_make_network_calls(self, monkeypatch):
        """
        _resolve_whisper_cache uses only os.path.isfile / os.path.isdir
        calls. If the cache is missing it returns 'missing' or 'unknown'
        without attempting any HTTP request.
        """
        import urllib.request
        original_urlopen = urllib.request.urlopen

        calls = []

        def tracking_urlopen(request, timeout=None):
            calls.append(request.full_url)
            return original_urlopen(request, timeout=timeout)

        monkeypatch.setattr(urllib.request, "urlopen", tracking_urlopen)
        monkeypatch.setenv("HF_HOME", r"E:\AI\huggingface")

        _resolve_whisper_cache("base.en")

        assert calls == [], (
            f"Expected no network calls but got: {calls}. "
            "_resolve_whisper_cache must not trigger any HTTP requests."
        )

    def test_resolve_missing_returns_missing_when_cache_dir_is_empty(self, monkeypatch):
        """
        When HF_HOME exists but the model snapshot is not there, status
        must be 'missing' — not 'unknown'.
        """
        # Use a real directory that exists but has no whisper cache
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_dir = os.path.join(tmpdir, "huggingface")
            os.makedirs(hf_dir)
            monkeypatch.setenv("HF_HOME", hf_dir)
            result = _resolve_whisper_cache("base.en")

            assert result["status"] == "missing", (
                f"Expected 'missing' for empty cache, got {result['status']!r}"
            )

    def test_resolve_unknown_when_hf_home_does_not_exist(self, monkeypatch):
        """
        When HF_HOME itself doesn't exist, status must be 'unknown' so the
        UI can distinguish a missing download from an unconfigured cache.
        """
        import uuid
        nonexistent = rf"C:\{uuid.uuid4().hex}"
        monkeypatch.setenv("HF_HOME", nonexistent)

        result = _resolve_whisper_cache("base.en")

        assert result["status"] == "unknown", (
            f"Expected 'unknown' for non-existent HF_HOME, got {result['status']!r}"
        )


# ---------------------------------------------------------------------------
# Test 2 — STT-unavailable state is handled gracefully
# ---------------------------------------------------------------------------

class TestSTTUnavailable:
    """
    When the whisper server is not ready, _is_stt_ready returns False,
    _transcribe returns empty, and VoiceListener never calls on_directed.
    The TTY remains usable throughout.
    """

    def test_is_stt_ready_false_when_no_server(self, monkeypatch):
        """
        Without a running whisper server and no getter set,
        _is_stt_ready must return False (not raise).
        """
        # Ensure no getter is registered
        set_stt_ready_getter(None)
        # Kill any real server that might be running so the test is deterministic
        if _is_whisper_running():
            pytest.skip("Whisper server is running — cannot test unavailable state")

        result = _is_stt_ready()
        assert result is False, (
            "_is_stt_ready() should return False when no server is running. "
            "Got True — a server is responding on port 9000."
        )

    def test_transcribe_returns_empty_when_not_ready(self, monkeypatch):
        """
        _transcribe must return '' when the STT server is unreachable,
        without raising any exception. The caller (_finalize) must then
        skip the on_directed callback.
        """
        # Mock _is_stt_ready to always return False
        def always_not_ready():
            return False

        monkeypatch.setattr(voice_listener, "_is_stt_ready", always_not_ready)

        # Even with valid PCM data, transcription must not crash
        pcm_frames = b"\x00\x00" * 16000  # 1 second of silence
        result = voice_listener._transcribe(pcm_frames)

        assert result == "", (
            f"_transcribe should return '' when STT is not ready, got {result!r}"
        )

    def test_voice_listener_does_not_crash_when_stt_unavailable(self, monkeypatch):
        """
        VoiceListener must instantiate and run its processing loop without
        crashing when the STT server is unreachable.
        """
        import queue
        import threading
        from unittest.mock import MagicMock

        # Pretend the server is never ready
        monkeypatch.setattr(voice_listener, "_is_stt_ready", lambda: False)

        directed = MagicMock()
        listener = VoiceListener(on_directed=directed)

        # Inject audio frames and let _process_loop run for 200ms
        stop = threading.Event()

        def feeder():
            # One full audio frame
            frame = b"\x00\x00" * 480  # 30ms of silence
            for _ in range(5):
                if stop.is_set():
                    break
                listener._audio_q.put(bytes(frame))
                time.sleep(0.04)
            listener._stop_event.set()

        t = threading.Thread(target=feeder, daemon=True)
        t.start()
        time.sleep(0.3)  # Let it process
        stop.set()
        t.join(timeout=2.0)

        # Must not have called on_directed — server was unavailable
        directed.assert_not_called(), (
            "on_directed should not be called when STT is unavailable, "
            f"but was called with: {directed.call_args_list}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Real-path load time from cache (integration)
# ---------------------------------------------------------------------------

class TestWhisperRealLoadTime:
    """
    Loads faster-whisper with the configured model from the real cache.
    Requires the model to be present on disk. Skips if cache is missing.
    """

    @pytest.mark.requires_network
    def test_model_loads_from_cache_under_30s(self):
        """
        The 'base.en' model must load from disk cache in under 30 seconds.
        On a system with the cache present, loading a CPU int8 model should
        be much faster — typically 2–10 s.
        """
        cache = _resolve_whisper_cache("base.en")
        if cache["status"] != "cached":
            pytest.skip(
                f"Model not cached at {cache['hf_home']!r} — "
                "run a real boot to populate the cache first."
            )

        # Import faster-whisper fresh (not from the venv — from wherever
        # the current Python can find it). If it's not installed in the
        # test environment this test gracefully skips.
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            pytest.skip("faster-whisper not installed in test environment")

        t0 = time.time()
        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        elapsed = time.time() - t0

        del model  # Free memory

        assert elapsed < 30.0, (
            f"Model loaded in {elapsed:.1f}s — expected < 30s. "
            "If this is the first load after a download, this is expected. "
            "Re-run after a successful boot to measure cached load time."
        )
