from __future__ import annotations

from scripts import run_wan_transition_9825


def test_build_parser_has_expected_flags():
    p = run_wan_transition_9825.build_parser()
    help_text = p.format_help()
    assert "--first" in help_text
    assert "--last" in help_text
    assert "--prompt" in help_text
    assert "--output" in help_text
    assert "--model-id" in help_text

