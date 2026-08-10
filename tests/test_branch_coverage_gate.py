import json
import subprocess
from types import SimpleNamespace

from scripts import check_branch_coverage


def test_branch_totals_require_branch_data():
    assert check_branch_coverage._branch_totals(
        {
            "meta": {"branch_coverage": True},
            "totals": {"covered_branches": 9, "num_branches": 10},
        }
    ) == (9, 10)

    for report in (
        {"meta": {"branch_coverage": False}},
        {"meta": {"branch_coverage": True}, "totals": {"num_branches": 0}},
    ):
        try:
            check_branch_coverage._branch_totals(report)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid coverage data should be rejected")


def test_coverage_json_uses_current_interpreter(monkeypatch):
    run = lambda *args, **kwargs: SimpleNamespace(  # noqa: E731
        stdout=json.dumps({"meta": {"branch_coverage": True}, "totals": {}})
    )
    monkeypatch.setattr(subprocess, "run", run)
    assert check_branch_coverage._coverage_json()["meta"]["branch_coverage"] is True


def test_main_passes_fails_and_reports_invalid_data(monkeypatch, capsys):
    monkeypatch.setattr(
        check_branch_coverage,
        "_coverage_json",
        lambda: {
            "meta": {"branch_coverage": True},
            "totals": {"covered_branches": 86, "num_branches": 100},
        },
    )
    assert check_branch_coverage.main(["--minimum", "85"]) == 0
    assert "86.00%" in capsys.readouterr().out
    assert check_branch_coverage.main(["--minimum", "90"]) == 1

    monkeypatch.setattr(
        check_branch_coverage,
        "_coverage_json",
        lambda: {"meta": {"branch_coverage": False}},
    )
    assert check_branch_coverage.main([]) == 2
    assert "Unable to verify" in capsys.readouterr().err
