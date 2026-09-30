#!/usr/bin/env python3
"""Entry point that works from any directory: python <repo>/preference/learner.py <command> ...
(the same commands as python -m alfapref).

The control center looks for the literal flag name "--assimilation" in this file to decide
whether the learner accepts plugin-assimilation features; keep it here if the flag is renamed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alfapref.cli import main  # noqa: E402

main()
