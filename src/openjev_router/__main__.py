"""`python -m openjev_router` — runs the CLI."""

from __future__ import annotations

import sys

from openjev_router.cli import main

if __name__ == "__main__":
    sys.exit(main())
