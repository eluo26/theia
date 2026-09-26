"""Allow `python -m vision` as well as `python -m vision.cli`."""

from vision.cli import main

raise SystemExit(main())
