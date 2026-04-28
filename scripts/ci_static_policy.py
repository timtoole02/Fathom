#!/usr/bin/env python3
"""Guard Fathom's default CI safety posture.

Default PR/push CI must stay offline/lightweight: no networked backend
acceptance smoke, no model downloads, and no non-default ONNX Runtime feature
binary path. This helper is intentionally small and dependency-free.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"


def main() -> None:
    text = CI.read_text(encoding="utf-8")
    failures: list[str] = []

    if re.search(r"cargo\s+test\b[^\n]*--features\s+[^\n]*onnx-embeddings-ort", text):
        failures.append("default CI must not run onnx-embeddings-ort feature tests")

    for line_number, line in enumerate(text.splitlines(), start=1):
        if "scripts/backend_acceptance_smoke.sh" not in line:
            continue
        stripped = line.strip()
        if stripped != "bash -n scripts/backend_acceptance_smoke.sh":
            failures.append(
                f"default CI must only syntax-check backend_acceptance_smoke.sh, line {line_number}: {stripped}"
            )

    if re.search(r"FATHOM_ACCEPTANCE_KEEP_ARTIFACTS|FATHOM_ACCEPTANCE_PORT|backend_acceptance_smoke\.sh\s*$", text):
        failures.append("default CI must not invoke networked backend acceptance smoke")

    if failures:
        raise SystemExit("CI static policy failed:\n- " + "\n- ".join(failures))

    print("CI static policy passed")


if __name__ == "__main__":
    main()
