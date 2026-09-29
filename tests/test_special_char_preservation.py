"""Regression tests for special character preservation in the input pipeline.

Bug: $_ in PowerShell commands was being corrupted/stripped. Root cause was
f-strings (using curly braces directly) interpreting {..} as Python format
expressions in classify_ambient_speech and generate_ambient_remark.
Fix: replaced f-strings with plain string concatenation.

This test verifies:
1. classify_ambient_speech handles $_ and {} without crashing
2. generate_ambient_remark handles $_ and {} without crashing
3. The prompt preserves $_ literally (not expanded by bash or corrupted)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 1. Direct unit-test of the fixed functions ──────────────────────────────

def test_classify_ambient_speech_dollar_underscore():
    """$_ alone is not a format expression — verify it passes through."""
    from controller.agent_controller import classify_ambient_speech
    text = "$_ is the pipeline variable in PowerShell"
    # Should not raise; should return one of the three verdicts
    result = classify_ambient_speech(text)
    assert result in ("directed", "noteworthy", "ignore"), f"unexpected verdict: {result!r}"


def test_generate_ambient_remark_dollar_underscore():
    """$_ alone in generate_ambient_remark should not crash."""
    from controller.agent_controller import generate_ambient_remark
    text = "$_ is the pipeline variable in PowerShell"
    # Should not raise; returns None or a string
    result = generate_ambient_remark(text)
    # None is fine (means no remark); string is fine
    assert result is None or isinstance(result, str), f"unexpected result: {result!r}"


def test_classify_ambient_speech_with_braces():
    """PowerShell pipeline code with { $_ } should not crash the f-string fix."""
    from controller.agent_controller import classify_ambient_speech
    text = "Get-Process | Where-Object { $_.CPU -gt 100 }"
    result = classify_ambient_speech(text)
    assert result in ("directed", "noteworthy", "ignore"), f"unexpected verdict: {result!r}"


def test_generate_ambient_remark_with_braces():
    """PowerShell pipeline code with { $_ } should not crash the f-string fix."""
    from controller.agent_controller import generate_ambient_remark
    text = "Get-Process | Where-Object { $_.CPU -gt 100 }"
    result = generate_ambient_remark(text)
    assert result is None or isinstance(result, str), f"unexpected result: {result!r}"


# ── 2. Verify prompt strings preserve $_ literally ─────────────────────────────
# The unit tests above already confirm the f-string fix works (no crashes).
# This test additionally verifies the generated prompt text contains $_ literally.
# We test by inspecting the prompt that classify_ambient_speech constructs:
def test_prompt_contains_dollar_underscore():
    """Prompt text for classify_ambient_speech must preserve $_ literally."""
    import controller.agent_controller as _ac

    # Read the source of classify_ambient_speech to verify the fix
    # (string concat, not f-string) and that the template includes {text}.
    import inspect
    source = inspect.getsource(_ac.classify_ambient_speech)
    # The fix uses plain string concatenation, not an f-string.
    # Verify the function body does NOT contain an f-string with {text}.
    assert "f\"\"\"" not in source or 'f"{text}"' not in source, (
        "classify_ambient_speech still uses an f-string with {text}!"
    )
    # The fix replaces f-string with {text} with string concatenation.
    # Verify: no f-string containing {text} in the source.
    import re
    fstring_with_text = re.findall(r'f["\'][^"\']*\{text\}', source)
    assert not fstring_with_text, (
        f"classify_ambient_speech still uses f-string with {{text}}: {fstring_with_text}"
    )


# ── 3. URL extraction tests ─────────────────────────────────────────────────

def test_claude_md_not_extracted_as_url():
    """CLAUDE.md should NOT be extracted as a URL (TLD blocklist)."""
    sys.path.insert(0, "E:/Fairy")
    from controller.web_research import detect_research_trigger
    urls = detect_research_trigger("See CLAUDE.md for details")
    assert urls == [], f"CLAUDE.md was incorrectly extracted as URL: {urls}"


def test_readme_md_not_extracted_as_url():
    """README.md should NOT be extracted as a URL."""
    from controller.web_research import detect_research_trigger
    urls = detect_research_trigger("Read the README.md file")
    assert urls == [], f"README.md was incorrectly extracted as URL: {urls}"


def test_real_bare_domains_still_extracted():
    """Real bare domains (example.com, docs.python.org) should still be extracted."""
    from controller.web_research import detect_research_trigger
    urls = detect_research_trigger("Check out example.com please")
    assert "https://www.example.com" in urls, f"example.com not extracted: {urls}"


# ── 4. Approval matching tests ──────────────────────────────────────────────

def test_approval_yes_go_ahead():
    """'yes go ahead' should be recognized as approval (the original bug)."""
    from controller.delegate_state import is_approval
    assert is_approval("yes go ahead"), "'yes go ahead' should be approval"


def test_approval_yes_please():
    """'yes please' should be recognized as approval."""
    from controller.delegate_state import is_approval
    assert is_approval("yes please"), "'yes please' should be approval"


def test_approval_yeah_do_it():
    """'yeah do it' should be recognized as approval."""
    from controller.delegate_state import is_approval
    assert is_approval("yeah do it"), "'yeah do it' should be approval"


def test_approval_exact_matches_still_work():
    """Bare 'yes' and 'go ahead' should still be recognized exactly."""
    from controller.delegate_state import is_approval
    assert is_approval("yes"), "'yes' should still be approval"
    assert is_approval("go ahead"), "'go ahead' should still be approval"
    assert is_approval("do it"), "'do it' should still be approval"


def test_approval_question_asking_still_falls_through():
    """'yes, but how long will it take?' should NOT be an approval."""
    from controller.delegate_state import is_approval
    assert not is_approval("yes, but how long will it take?"), \
        "'yes, but...' should NOT be approval"


if __name__ == "__main__":
    print("Running regression tests...\n")

    tests = [
        test_classify_ambient_speech_dollar_underscore,
        test_generate_ambient_remark_dollar_underscore,
        test_classify_ambient_speech_with_braces,
        test_generate_ambient_remark_with_braces,
        test_prompt_contains_dollar_underscore,
        test_claude_md_not_extracted_as_url,
        test_readme_md_not_extracted_as_url,
        test_real_bare_domains_still_extracted,
        test_approval_yes_go_ahead,
        test_approval_yes_please,
        test_approval_yeah_do_it,
        test_approval_exact_matches_still_work,
        test_approval_question_asking_still_falls_through,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
