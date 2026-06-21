"""Backward-compatible re-exports — see lib/failure_analysis.py."""

from lib.failure_analysis import analyze_failure, hints_for_result, hints_for_text

__all__ = ["analyze_failure", "hints_for_result", "hints_for_text"]
