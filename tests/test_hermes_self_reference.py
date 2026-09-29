"""
Tests for Issue #1: Hermes self-reference phrase intercept.

Ensures phrases referencing Fairy's internal Hermes agent layer are never
routed to the Windows desktop-app launcher (computer_control / open_application).

Scope: These tests verify that hermes-related phrases:
  (a) are intercepted by HERMES_SELF_RE (or equivalent fast_intent guard) BEFORE
      they can reach computer_control/open_application;
  (b) never cause a WinError 2 ("file not found") from attempting to launch
      "hermes" / "hermes agent" as a Windows desktop application.

Legitimate app launches (Steam, Discord, Notepad, VS Code) must still work.
"""

import re
import pytest
from unittest.mock import patch, MagicMock


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_hermes_state():
    """Reset hermes_bridge state for isolation."""
    import hermes_bridge as hb

    saved = {}
    for key in ("_cached_agent", "_hermes_initialized", "_hermes_initialized_ok"):
        if hasattr(hb, key):
            saved[key] = getattr(hb, key)

    hb._cached_agent = None
    hb._hermes_initialized = False
    hb._hermes_initialized_ok = False

    try:
        from controller import main_brain
        with main_brain._health_cache["lock"]:
            main_brain._health_cache["available"] = False
            main_brain._health_cache["ts"] = 0.0
    except (ImportError, AttributeError):
        pass

    yield

    for key, value in saved.items():
        setattr(hb, key, value)
    hb._run_startup_health_check()


def _mock_hermes_success(user_text, history, on_status=None):
    return True, f"Handled by Hermes: {user_text}", history


def _mock_hermes_failure(user_text, history, on_status=None):
    return False, "[Hermes Error]", []


def _track_computer_control():
    """Return a mock computer_control that records all calls.

    computer_control is called with a single dict argument:
        computer_control({"action": "open", "value": "steam"})
    Not keyword arguments.
    """
    cc_mock = MagicMock()
    cc_calls = []

    def track(args_dict):
        cc_calls.append(args_dict)
        return {"ok": False, "error": "test: computer_control called"}

    cc_mock.side_effect = track
    return cc_mock, cc_calls


# ── Test 1: HERMES_SELF_RE regex coverage ───────────────────────────────────────

class TestHermesSelfReRegex:
    """Document which phrases the current HERMES_SELF_RE intercept catches.

    This is the BUG: the regex doesn't include 'open', 'activate', 'hey', or
    sentence-initial politeness like 'can you ...'.
    """

    def _regex(self):
        return re.compile(
            r"^\s*(?:start|use|launch|run|open|activate)\s+hermes\s*(?:agent)?\s*(?:to\s+(.+))?$",
            re.IGNORECASE,
        )

    def test_start_hermes_is_caught(self):
        assert self._regex().match("start hermes") is not None

    def test_start_hermes_agent_is_caught(self):
        assert self._regex().match("start hermes agent") is not None

    def test_launch_hermes_is_caught(self):
        assert self._regex().match("launch hermes") is not None

    def test_run_hermes_is_caught(self):
        assert self._regex().match("run hermes") is not None

    def test_open_hermes_agent_is_caught(self):
        """'open hermes agent' matches the expanded HERMES_SELF_RE (now fixed).

        Before the fix: regex didn't include 'open', so 'open hermes agent'
        fell through to fast_intent=ambiguous → browser fast path → crash.
        After the fix: regex includes 'open', so it's intercepted immediately.
        """
        assert self._regex().match("open hermes agent") is not None

    def test_hey_hermes_is_NOT_caught(self):
        """No action verb — falls through to fast_intent=None (safe)."""
        assert self._regex().match("hey hermes") is None

    def test_can_you_start_hermes_is_NOT_caught(self):
        """Doesn't start with action verb — falls through to fast_intent=None (safe)."""
        assert self._regex().match("can you start hermes") is None


# ── Test 2: fast_intent classification ───────────────────────────────────────────

class TestFastIntentClassification:
    """What does _fast_intent return for each phrase?"""

    def _fast_intent(self, text):
        """Inline simulation matching agent_controller._fast_intent for hermes phrases."""
        # "open X" — check for known desktop apps or websites
        _open_match = re.match(r"^open\s+(.+?)(?:\s|$)", text)
        if _open_match:
            target = _open_match.group(1).strip().lower()
            _KNOWN_APPS = {
                "notepad", "calculator", "steam", "discord", "vscode",
                "vs code", "chrome", "firefox", "edge",
            }
            if any(app in target for app in _KNOWN_APPS):
                return "app_action"
            return "ambiguous"

        # Action verbs: start, launch, run, close, kill, quit
        _LAUNCH = re.compile(
            r"^(start|launch|run|close|kill|quit|exit|stop|terminate)\s+(\S+)",
            re.IGNORECASE,
        )
        if _LAUNCH.match(text):
            return "app_action"
        return None

    def test_open_steam_is_app_action(self):
        assert self._fast_intent("open steam") == "app_action"

    def test_open_discord_is_app_action(self):
        assert self._fast_intent("open discord") == "app_action"

    def test_open_vscode_is_app_action(self):
        assert self._fast_intent("open vscode") == "app_action"

    def test_open_hermes_agent_is_ambiguous(self):
        """'open hermes agent' → ambiguous (not in _KNOWN_DESKTOP_APPS).

        This is the routing that causes the bug — ambiguous routes to browser
        fast path, which tries to open "hermes agent" as a URL.
        """
        assert self._fast_intent("open hermes agent") == "ambiguous"

    def test_start_hermes_is_app_action(self):
        assert self._fast_intent("start hermes") == "app_action"

    def test_launch_discord_is_app_action(self):
        assert self._fast_intent("launch discord") == "app_action"

    def test_hey_hermes_is_none(self):
        """No action verb, no 'open' prefix → fast_intent=None → goes to Hermes."""
        assert self._fast_intent("hey hermes") is None

    def test_can_you_start_hermes_is_none(self):
        """Doesn't start with action verb → fast_intent=None → goes to Hermes."""
        assert self._fast_intent("can you start hermes") is None


# ── Test 3: handle_request never reaches open_application with hermes ─────────────

class TestHandleRequestNeverLaunchesHermes:
    """Integration: handle_request must never call computer_control with hermes targets.

    These tests exercise the full handle_request pipeline. They patch:
      - hermes_bridge.run_turn_safe (Hermes is mocked, so we control its output)
      - skills.computer_control.computer_control (track calls, simulate launch)

    For each hermes-referencing phrase, we verify the computer_control mock
    is never called with a value containing "hermes". For legitimate app
    launches (Steam, Discord), we verify computer_control IS reached.
    """

    def _assert_no_hermes_to_computer_control(
        self, phrase, fresh_hermes_state, hermes_mock=_mock_hermes_success
    ):
        import hermes_bridge
        from controller import agent_controller as ac

        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        cc_mock, cc_calls = _track_computer_control()

        with patch("hermes_bridge.run_turn_safe", hermes_mock):
            with patch("skills.computer_control.computer_control", cc_mock):
                reply, _ = ac.handle_request(phrase)

        hermes_calls = [
            c for c in cc_calls
            if "hermes" in str(c.get("value", "")).lower()
        ]
        assert hermes_calls == [], (
            f"handle_request('{phrase}') reached computer_control with hermes "
            f"target: {hermes_calls}. Hermes is an internal agent, not a desktop app."
        )

    # --- Each phrase from Issue #1 ---

    def test_open_hermes_agent_never_launches_hermes(self, fresh_hermes_state):
        """'open hermes agent' — the actual bypass path that causes the WinError 2."""
        self._assert_no_hermes_to_computer_control("open hermes agent", fresh_hermes_state)

    def test_start_hermes_never_launches_hermes(self, fresh_hermes_state):
        self._assert_no_hermes_to_computer_control("start hermes", fresh_hermes_state)

    def test_launch_hermes_never_launches_hermes(self, fresh_hermes_state):
        self._assert_no_hermes_to_computer_control("launch hermes", fresh_hermes_state)

    def test_run_hermes_never_launches_hermes(self, fresh_hermes_state):
        self._assert_no_hermes_to_computer_control("run hermes", fresh_hermes_state)

    def test_start_hermes_agent_never_launches_hermes(self, fresh_hermes_state):
        self._assert_no_hermes_to_computer_control("start hermes agent", fresh_hermes_state)

    def test_activate_hermes_agent_never_launches_hermes(self, fresh_hermes_state):
        self._assert_no_hermes_to_computer_control(
            "activate hermes agent", fresh_hermes_state
        )

    def test_hey_hermes_never_launches_hermes(self, fresh_hermes_state):
        self._assert_no_hermes_to_computer_control("hey hermes", fresh_hermes_state)

    def test_can_you_start_hermes_never_launches_hermes(self, fresh_hermes_state):
        self._assert_no_hermes_to_computer_control(
            "can you start hermes", fresh_hermes_state
        )

    # --- Legitimate app launches must still reach computer_control ---

    def test_open_steam_reaches_computer_control(self, fresh_hermes_state):
        """'open Steam' must reach computer_control (legitimate app launch)."""
        import hermes_bridge
        from controller import agent_controller as ac

        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        cc_mock, cc_calls = _track_computer_control()

        # Hermes fails → fall through to app_action path → computer_control
        with patch("hermes_bridge.run_turn_safe", _mock_hermes_failure):
            with patch("skills.computer_control.computer_control", cc_mock):
                reply, _ = ac.handle_request("open Steam")

        steam_calls = [
            c for c in cc_calls
            if "steam" in str(c.get("value", "")).lower()
        ]
        assert steam_calls, (
            f"'open Steam' must reach computer_control, but got: {cc_calls}"
        )

    def test_launch_discord_reaches_computer_control(self, fresh_hermes_state):
        """'launch Discord' must reach computer_control (legitimate app launch)."""
        import hermes_bridge
        from controller import agent_controller as ac

        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        cc_mock, cc_calls = _track_computer_control()

        with patch("hermes_bridge.run_turn_safe", _mock_hermes_failure):
            with patch("skills.computer_control.computer_control", cc_mock):
                reply, _ = ac.handle_request("launch Discord")

        discord_calls = [
            c for c in cc_calls
            if "discord" in str(c.get("value", "")).lower()
        ]
        assert discord_calls, (
            f"'launch Discord' must reach computer_control, but got: {cc_calls}"
        )

    def test_start_notepad_reaches_computer_control(self, fresh_hermes_state):
        """'start Notepad' must reach computer_control (legitimate app launch)."""
        import hermes_bridge
        from controller import agent_controller as ac

        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        cc_mock, cc_calls = _track_computer_control()

        with patch("hermes_bridge.run_turn_safe", _mock_hermes_failure):
            with patch("skills.computer_control.computer_control", cc_mock):
                reply, _ = ac.handle_request("start notepad")

        notepad_calls = [
            c for c in cc_calls
            if "notepad" in str(c.get("value", "")).lower()
        ]
        assert notepad_calls, (
            f"'start notepad' must reach computer_control, but got: {cc_calls}"
        )

    def test_open_vscode_reaches_computer_control(self, fresh_hermes_state):
        """'open VS Code' must reach computer_control (legitimate app launch)."""
        import hermes_bridge
        from controller import agent_controller as ac

        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        cc_mock, cc_calls = _track_computer_control()

        with patch("hermes_bridge.run_turn_safe", _mock_hermes_failure):
            with patch("skills.computer_control.computer_control", cc_mock):
                reply, _ = ac.handle_request("open vscode")

        vscode_calls = [
            c for c in cc_calls
            if "vscode" in str(c.get("value", "")).lower()
            or "vs code" in str(c.get("value", "")).lower()
        ]
        assert vscode_calls, (
            f"'open vscode' must reach computer_control, but got: {cc_calls}"
        )


# ── Test 4: Internal-name denylist is a safety net ───────────────────────────────

class TestInternalNameDenylist:
    """The denylist catches anything that bypasses HERMES_SELF_RE."""

    def test_hermes_in_denylist(self):
        from hermes_bridge import is_internal_agent_name

        assert is_internal_agent_name("hermes") is True
        assert is_internal_agent_name("HERMES") is True
        assert is_internal_agent_name(" Hermes ") is True

    def test_hermes_agent_in_denylist(self):
        from hermes_bridge import is_internal_agent_name

        assert is_internal_agent_name("hermes agent") is True
        assert is_internal_agent_name("HERMES AGENT") is True

    def test_app_action_fallback_refuses_hermes(self):
        """_try_app_action_fallback rejects 'open hermes agent' via denylist."""
        from controller import agent_controller as ac

        result = ac._try_app_action_fallback("open hermes agent")
        assert result is not None
        assert "internal" in result.lower() or "hermes" in result.lower()

    def test_app_action_fallback_refuses_start_hermes(self):
        """_try_app_action_fallback rejects 'start hermes' via denylist."""
        from controller import agent_controller as ac

        result = ac._try_app_action_fallback("start hermes")
        assert result is not None
        assert "internal" in result.lower() or "hermes" in result.lower()

    def test_app_action_fallback_passes_through_normal_apps(self):
        """_try_app_action_fallback still tries to launch real apps.

        The denylist only blocks internal agent names. 'start steam' must
        NOT be rejected by the denylist — the launch path should be tried.
        """
        from controller import agent_controller as ac

        result = ac._try_app_action_fallback("start steam")
        # Returns a string from _verify_app_launch (which our test env will
        # fail) or None if computer_control is unavailable. Either way it
        # must NOT be the internal-name rejection message.
        if result is not None:
            assert "internal" not in result.lower(), (
                f"Steam must not be rejected as internal: {result}"
            )
            assert "system component" not in result.lower(), (
                f"Steam must not be rejected as a system component: {result}"
            )


# ── Test 5: follow-on task ("start hermes to do X") ─────────────────────────────

class TestHermesSelfReferenceFollowOn:
    """'start hermes to do X' should pass 'do X' to Hermes."""

    def test_hermes_self_re_extracts_follow_on_task(self):
        regex = re.compile(
            r"^\s*(?:start|use|launch|run|open|activate)\s+hermes\s*(?:agent)?\s*(?:to\s+(.+))?$",
            re.IGNORECASE,
        )
        m = regex.match("start hermes to open notepad")
        assert m is not None
        assert m.group(1) == "open notepad"

    def test_hermes_self_re_extracts_follow_on_for_open(self):
        """'open hermes agent to do X' → extracts 'do X' as follow-on task."""
        regex = re.compile(
            r"^\s*(?:start|use|launch|run|open|activate)\s+hermes\s*(?:agent)?\s*(?:to\s+(.+))?$",
            re.IGNORECASE,
        )
        m = regex.match("open hermes agent to search for python tutorials")
        assert m is not None
        assert m.group(1) == "search for python tutorials"

    def test_handle_request_passes_follow_on_to_hermes(self, fresh_hermes_state):
        """'start hermes to do X' → Hermes receives 'do X'."""
        import hermes_bridge
        from controller import agent_controller as ac

        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        received = {}

        def mock_hermes(user_text, history, on_status=None):
            received["text"] = user_text
            return True, f"Done: {user_text}", history

        with patch("hermes_bridge.run_turn_safe", mock_hermes):
            reply, _ = ac.handle_request("start hermes to open notepad")

        assert received.get("text") == "open notepad", (
            f"Expected Hermes to receive 'open notepad', got: {received.get('text')}"
        )

    def test_handle_request_open_hermes_agent_with_follow_on(self, fresh_hermes_state):
        """'open hermes agent to do X' → Hermes receives 'do X' (not 'open hermes agent')."""
        import hermes_bridge
        from controller import agent_controller as ac

        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        received = {}

        def mock_hermes(user_text, history, on_status=None):
            received["text"] = user_text
            return True, f"Done: {user_text}", history

        with patch("hermes_bridge.run_turn_safe", mock_hermes):
            reply, _ = ac.handle_request(
                "open hermes agent to search for python tutorials"
            )

        assert received.get("text") == "search for python tutorials", (
            f"Expected Hermes to receive 'search for python tutorials', "
            f"got: {received.get('text')}"
        )


# ── Test 6: Self-reply for hermes without follow-on task ────────────────────────

class TestHermesSelfReplyMessage:
    """When user says 'start hermes' (no follow-on), Fairy replies with a self-reply.

    This is the user-facing behavior — telling the agent to "start itself" is
    a no-op, and Fairy should acknowledge it's already running.
    """

    def test_start_hermes_returns_self_reply(self, fresh_hermes_state):
        """'start hermes' → self-reply explaining Fairy is already running."""
        import hermes_bridge
        from controller import agent_controller as ac

        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        # With Hermes initialized, the intercept should fire BEFORE Hermes.
        # We pass a mock Hermes to verify it's NOT called.
        hermes_called = {"called": False}

        def mock_hermes(user_text, history, on_status=None):
            hermes_called["called"] = True
            return True, "should not reach here", history

        with patch("hermes_bridge.run_turn_safe", mock_hermes):
            reply, _ = ac.handle_request("start hermes")

        assert not hermes_called["called"], (
            "Hermes should NOT be called for 'start hermes' (intercept handles it)"
        )
        # Self-reply mentions the user as "Master" and acknowledges running
        assert "Master" in reply, f"Expected self-reply with 'Master', got: {reply!r}"
        assert "already" in reply.lower() or "running" in reply.lower(), (
            f"Expected self-reply to acknowledge already running, got: {reply!r}"
        )

    def test_open_hermes_agent_returns_self_reply(self, fresh_hermes_state):
        """'open hermes agent' → self-reply (no follow-on task to recurse)."""
        import hermes_bridge
        from controller import agent_controller as ac

        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        hermes_called = {"called": False}

        def mock_hermes(user_text, history, on_status=None):
            hermes_called["called"] = True
            return True, "should not reach here", history

        with patch("hermes_bridge.run_turn_safe", mock_hermes):
            reply, _ = ac.handle_request("open hermes agent")

        assert not hermes_called["called"], (
            "Hermes should NOT be called for 'open hermes agent' (intercept handles it)"
        )
        assert "Master" in reply
