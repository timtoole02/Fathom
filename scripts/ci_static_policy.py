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

    saw_public_contract_smoke = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if "scripts/backend_acceptance_smoke.sh" in line:
            if stripped != "bash -n scripts/backend_acceptance_smoke.sh":
                failures.append(
                    f"default CI must only syntax-check backend_acceptance_smoke.sh, line {line_number}: {stripped}"
                )
        if "scripts/public_api_contract_smoke.sh" in line:
            if stripped in {"bash scripts/public_api_contract_smoke.sh", "run: bash scripts/public_api_contract_smoke.sh"}:
                saw_public_contract_smoke = True
            elif stripped not in {"bash -n scripts/public_api_contract_smoke.sh", "run: bash -n scripts/public_api_contract_smoke.sh"}:
                failures.append(
                    f"default CI may only syntax-check or run public_api_contract_smoke.sh, line {line_number}: {stripped}"
                )

    if not saw_public_contract_smoke:
        failures.append("default CI must run the no-download public API contract smoke")

    if re.search(r"FATHOM_ACCEPTANCE_KEEP_ARTIFACTS|FATHOM_ACCEPTANCE_PORT|backend_acceptance_smoke\.sh\s*$", text):
        failures.append("default CI must not invoke networked backend acceptance smoke")

    if "FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR" in text:
        failures.append("default CI must not persist public contract smoke artifacts")

    if re.search(r"actions/(upload-artifact|cache)@", text):
        failures.append("default CI must not upload or cache QA/model artifacts")

    if re.search(r"(models|artifacts|target)\s*:", text, re.IGNORECASE):
        failures.append("default CI must not define broad model/artifact/target caches")

    if failures:
        raise SystemExit("CI static policy failed:\n- " + "\n- ".join(failures))

    print("CI static policy passed")


if __name__ == "__main__":
    main()
