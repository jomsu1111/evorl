"""EvoRL: learning where to store experience (memory vs. parameters) with RL.

COVE's source tree (../cove/src) is put on sys.path so its memory manager,
prompts and task templates can be reused unchanged.
"""
import os
import sys

_COVE_SRC = os.environ.get("COVE_SRC") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cove", "src")
if _COVE_SRC not in sys.path:
    sys.path.insert(0, _COVE_SRC)
