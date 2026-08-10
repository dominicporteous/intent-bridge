"""Run the opt-in endpoint/LLM/tool benchmark with configuration from .env."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    # This must precede imports of intent_bridge.config, whose settings object is
    # intentionally constructed once at import time.
    load_dotenv()

    from benchmark.full_pipeline import FullPipelineBenchmarkMatcher
    from benchmark.loader import load_corpus
    from benchmark.runner import compare_operations

    home_filters = {
        item.strip()
        for item in os.environ.get("BENCHMARK_HOME", "").split(",")
        if item.strip()
    }
    source_filters = {
        item.strip()
        for item in os.environ.get("BENCHMARK_SOURCE", "").split(",")
        if item.strip()
    }
    limit = max(1, int(os.environ.get("BENCHMARK_LIMIT", "25")))
    force_llm = _enabled("BENCHMARK_FORCE_LLM")

    examples = [
        example
        for example in load_corpus().examples
        if (not home_filters or example.request.home.home_id in home_filters)
        and (
            not source_filters
            or any(source in example.diagnostic_id for source in source_filters)
        )
    ][:limit]
    if not examples:
        print("No benchmark examples matched BENCHMARK_HOME/BENCHMARK_SOURCE.")
        return 2

    async def run() -> int:
        matcher = FullPipelineBenchmarkMatcher(force_llm=force_llm)
        failures = 0
        route_counts: dict[str, int] = {}
        for index, example in enumerate(examples, 1):
            try:
                result = await matcher.match(example.request)
                missing, unexpected = compare_operations(example.expected, result.operations)
            except Exception as exc:
                failures += 1
                print(
                    f"FAIL {index}/{len(examples)} {example.diagnostic_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            for route in matcher.last_routes:
                route_counts[route] = route_counts.get(route, 0) + 1
            if missing or unexpected:
                failures += 1
                print(f"FAIL {index}/{len(examples)} {example.diagnostic_id}")
                print(f"  routes={matcher.last_routes!r}")
                print(f"  missing={missing!r}")
                print(f"  unexpected={unexpected!r}")
                print(f"  response={result.response!r}")
            else:
                print(
                    f"PASS {index}/{len(examples)} {example.diagnostic_id} "
                    f"routes={matcher.last_routes!r}"
                )
        passed = len(examples) - failures
        print(
            f"Full pipeline: {passed}/{len(examples)} passed; "
            f"force_llm={force_llm}; routes={route_counts}"
        )
        return 0 if failures == 0 else 1

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
