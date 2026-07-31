#!/usr/bin/env python3
"""Memory bench (S15.2).

Idle RSS targets per architecture §5.7:
  * macOS  : < 200 MB
  * Linux  : < 200 MB (sidecars not yet built)
  * Windows: < 300 MB (sidecars not yet built)

Reads /proc/self/status (Linux) or `ps` (macOS/BSD) and prints a JSON
record. Doesn't fail the build above the budget — it's tracked as a
trend, alerted on regressions in CI.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys


def _rss_bytes() -> int:
    if sys.platform == "linux":
        # ru_maxrss is in KB on Linux.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if sys.platform == "darwin":
        # ru_maxrss is in bytes on macOS.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Fallback: shell out to ps.
    out = subprocess.check_output(  # noqa: S603
        ["ps", "-o", "rss=", "-p", str(os.getpid())]
    )
    return int(out.strip()) * 1024


def main() -> None:
    # Touch the heavy imports so the measurement is realistic.
    import meetmind  # noqa: F401
    from meetmind.api.bus import EventBus  # noqa: F401
    from meetmind.api.coach import CoachLoop  # noqa: F401
    from meetmind.memory.store import Store  # noqa: F401

    rss = _rss_bytes()
    rss_mb = round(rss / (1024 * 1024), 1)
    budget = 200 if sys.platform != "win32" else 300
    print(
        json.dumps(
            {
                "metric": "idle_rss_mb",
                "value": rss_mb,
                "budget_mb": budget,
                "platform": platform.platform(),
                "within_budget": rss_mb < budget,
            }
        )
    )


if __name__ == "__main__":
    main()
