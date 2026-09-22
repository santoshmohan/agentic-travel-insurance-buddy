"""Compatibility package for the legacy eval import path.

This project stores the evaluation modules under the `evals/` directory, but the
existing tests and code import them as `eval.*`. The wrappers here re-export the
real implementations from the canonical `evals` package so both import paths work.
"""

from evals import *  # noqa: F401,F403
