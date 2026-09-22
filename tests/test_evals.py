from __future__ import annotations

import json
from pathlib import Path

from skillroute.catalog import Catalog
from skillroute.evals import run_golden_routes
from skillroute.routing import Router


def test_golden_route_eval_passes(indexed_catalog: Catalog) -> None:
    cases = Path(__file__).parent / "fixtures" / "golden_routes.json"

    results = run_golden_routes(Router(indexed_catalog), cases)

    assert len(results) == 4
    assert all(result.passed for result in results)


def test_golden_route_eval_null_clarification_skips_flag(indexed_catalog: Catalog, tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([
        {"request": "no skills match this at all xyzzy",
         "expected_skill_names": [],
         "expect_clarification": None},
    ]))

    results = run_golden_routes(Router(indexed_catalog), cases_path)

    # clarification_needed is True for a no-match query, but the explicit
    # null means the flag is not asserted — only skill recall is measured.
    assert len(results) == 1
    assert results[0].passed is True
    assert not results[0].notes

