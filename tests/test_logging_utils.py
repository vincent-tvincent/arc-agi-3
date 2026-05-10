from __future__ import annotations

import csv

from arc_agent.training.logging_utils import RunLogger


def test_run_logger_expands_csv_fields_for_new_metric_keys(tmp_path):
    logger = RunLogger("test_run_logger_expands_csv_fields", tmp_path, use_csv=True, use_tensorboard=False)

    logger.log_metrics("progress", 1, {"loss": 0.5})
    logger.log_metrics("epoch", 2, {"val/loss": 0.4, "val/mean_f1": 0.2})
    logger.close()

    with (tmp_path / "metrics.csv").open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["loss"] == "0.5"
    assert rows[0]["val/loss"] == ""
    assert rows[1]["val/loss"] == "0.4"
    assert rows[1]["val/mean_f1"] == "0.2"
