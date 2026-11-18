"""The skill package must lint cleanly (name, size limits, forbidden patterns)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skill")))

import build_skill  # noqa: E402


def test_skill_package_lints_clean_and_script_fits_limit():
    files = [(os.path.join(build_skill.NAME, os.path.basename(p)), p)
             for p in (build_skill.SCRIPT, os.path.join(build_skill.SKILL_DIR, "SKILL.md"))]
    assert build_skill.lint(files) == []
    assert os.path.getsize(build_skill.SCRIPT) <= build_skill.MAX_SCRIPT_BYTES
