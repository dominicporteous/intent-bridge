"""Process setup for executable benchmark runs."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Application settings are immutable after their first import. Load repository
# configuration before pytest imports benchmark modules and their adapters.
load_dotenv()


def pytest_configure(config):
    """Keep benchmark temp files out of stale or ACL-restricted user temp roots."""

    if config.option.basetemp is not None:
        return
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    config.option.basetemp = str(
        Path(__file__).resolve().parent.parent / ".cache" / f"pytest-benchmark-{worker_id}"
    )
