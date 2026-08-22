"""
Pytest configuration and shared fixtures for LLM Gateway test suite.
"""

import pytest
import sys
from pathlib import Path

# Ensure src/ is on python path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
