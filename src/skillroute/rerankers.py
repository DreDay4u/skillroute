from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Protocol

from skillroute.models import RouteCandidate, to_jsonable

# Fleet default (LeTrain): when SKILLROUTE_RERANKER_CMD is unset and the
# shared Jev re-ranker is installed, use it. Set the env var to empty ("")
# to force the heuristic reranker (hermetic tests / offline runs).
FLEET_RERANKER = str(Path.home() / ".hermes/shared/scripts/jev-skill-reranker.py")


class Reranker(Protocol):
    def rerank(self, request: str, candidates: list[RouteCandidate]) -> list[RouteCandidate]:
        ...


class HeuristicReranker:
    name = "heuristic"

    def rerank(self, request: str, candidates: list[RouteCandidate]) -> list[RouteCandidate]:
        return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)


class ExternalCommandReranker:
    """LLM-compatible reranker hook using a local command contract.

    The command receives JSON on stdin with the request and candidates, and must
    return JSON with `ordered_skill_ids`. This keeps the core provider-neutral
    while allowing OpenAI, local models, or custom rerankers to plug in later.
    """

    def __init__(self, command: str) -> None:
        self.command = command

    @property
    def name(self) -> str:
        return "jev" if "jev-skill-reranker" in self.command else "external"

    def rerank(self, request: str, candidates: list[RouteCandidate]) -> list[RouteCandidate]:
        argv = shlex.split(self.command)
        if not argv:
            return candidates
        payload = {"request": request, "candidates": to_jsonable(candidates)}
        try:
            completed = subprocess.run(
                argv,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            return candidates
        if completed.returncode != 0:
            return candidates
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return candidates
        ordered_ids = result.get("ordered_skill_ids")
        if not isinstance(ordered_ids, list):
            return candidates
        by_id = {candidate.skill_id: candidate for candidate in candidates}
        ordered = [by_id[skill_id] for skill_id in ordered_ids if skill_id in by_id]
        ordered.extend(candidate for candidate in candidates if candidate.skill_id not in ordered_ids)
        return ordered


def default_reranker() -> Reranker:
    if "SKILLROUTE_RERANKER_CMD" in os.environ:
        command = os.environ["SKILLROUTE_RERANKER_CMD"]
        if command:
            return ExternalCommandReranker(command)
        return HeuristicReranker()  # explicitly disabled (empty value)
    if os.path.isfile(FLEET_RERANKER):
        return ExternalCommandReranker(f"python3 {shlex.quote(FLEET_RERANKER)}")
    return HeuristicReranker()

