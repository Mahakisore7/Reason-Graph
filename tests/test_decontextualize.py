"""Tests for parsing the stand-alone rewrite -- no model needed."""

from __future__ import annotations

from scripts.decontextualize_felm import parse_standalone

ORIGINAL = "They are typically located on the wings."


def test_reads_the_rewritten_sentence() -> None:
    output = "Standalone sentence: Ground spoilers are typically located on the wings."
    assert (
        parse_standalone(output, ORIGINAL) == "Ground spoilers are typically located on the wings."
    )


def test_tolerates_bold_markers_and_extra_lines() -> None:
    output = "Sure!\n**Standalone sentence:** Ground spoilers sit on the wings.\nDone."
    assert parse_standalone(output, ORIGINAL) == "Ground spoilers sit on the wings."


def test_falls_back_to_original_when_format_is_ignored() -> None:
    assert parse_standalone("Ground spoilers are on the wings.", ORIGINAL) == ORIGINAL
    assert parse_standalone("Standalone sentence:", ORIGINAL) == ORIGINAL


def test_rewrite_must_entail_the_original() -> None:
    from scripts.decontextualize_felm import keep_if_faithful
    from src.tree.nli_labeler import FakeDirectionalNLI

    good = "Ground spoilers are typically located on the wings."
    dropped = "Ground spoilers exist."
    nli = FakeDirectionalNLI(labels={(good, ORIGINAL): "entailment"})
    assert keep_if_faithful(good, ORIGINAL, nli) == good
    assert keep_if_faithful(dropped, ORIGINAL, nli) == ORIGINAL
    assert keep_if_faithful(ORIGINAL, ORIGINAL, nli) == ORIGINAL
