"""Test URL extraction fix - debug."""
import sys
sys.path.insert(0, "E:/Fairy")

from controller.web_research import detect_research_trigger

test_cases = [
    # Should be detected
    ("Check out https://example.com/guide", ["https://example.com/guide"]),
    ("Read https://docs.python.org/3/library/", ["https://docs.python.org/3/library/"]),
    ("Look at example.com please", ["https://www.example.com"]),
    ("See www.google.com", ["https://www.google.com"]),
    ("Visit docs.python.org", ["https://www.docs.python.org"]),
    ("Go to github.com/user/repo", ["https://www.github.com/user/repo"]),
    # Should NOT be detected (file paths, not URLs)
    ("See CLAUDE.md", []),
    ("Read README.md for details", []),
    ("Check the config.py file", []),
    ("See main.py", []),
    ("Look at data.json", []),
    ("The settings.yaml file", []),
    ("Read the docs at README.md", []),
    ("Check package.json", []),
    ("View index.html", []),
    ("Read CONTRIBUTING.md", []),
    # Mixed
    ("Visit example.com and read CLAUDE.md", ["https://www.example.com"]),
    ("See https://github.com and CLAUDE.md", ["https://github.com"]),
    # Edge cases
    ("example.org", ["https://www.example.org"]),
    ("the.md", []),  # single label, no real TLD
    ("foo.bar", ["https://www.foo.bar"]),  # bar is a real TLD
]

all_pass = True
for text, expected in test_cases:
    result = detect_research_trigger(text)
    status = "PASS" if result == expected else "FAIL"
    if result != expected:
        all_pass = False
        print(f"[{status}] {repr(text)[:50]:50} -> {result}")
        print(f"  expected: {expected}")
    else:
        print(f"[{status}] {repr(text)[:50]:50} -> {result}")

print()
print("All tests passed!" if all_pass else "SOME TESTS FAILED!")
