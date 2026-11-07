from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import load_prepared, prepare_lendingclub
from .part_a import run_part_a
from .part_b import run_part_b
from .reporting import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-part LendingClub selective-label CVaR evaluation")
    parser.add_argument("stage", choices=["prepare", "part-a", "part-b", "report", "evaluate", "all"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--accepted", help="Override accepted-loan CSV/CSV.GZ path")
    parser.add_argument("--rejected", help="Override rejected-application CSV/CSV.GZ path")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Override XGBoost device")
    parser.add_argument("--repetitions", type=int, help="Override Part B repetitions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, _ = load_config(args.config)
    if args.accepted:
        cfg["data"]["accepted_path"] = str(Path(args.accepted).expanduser().resolve())
    if args.rejected:
        cfg["data"]["rejected_path"] = str(Path(args.rejected).expanduser().resolve())
    if args.device:
        cfg["xgboost"]["device"] = args.device
    if args.repetitions is not None:
        if args.repetitions < 1:
            raise ValueError("--repetitions must be positive")
        cfg["part_b"]["repetitions"] = args.repetitions

    if args.stage in {"prepare", "all"}:
        data = prepare_lendingclub(cfg)
        print(f"Prepared {len(data):,} applications at {cfg['data']['prepared_path']}")
    else:
        data = None
    if args.stage in {"part-a", "part-b", "evaluate", "all"} and data is None:
        data = load_prepared(cfg)
    if args.stage in {"part-a", "evaluate", "all"}:
        run_part_a(data, cfg)
        print("Part A complete")
    if args.stage in {"part-b", "evaluate", "all"}:
        run_part_b(data, cfg)
        print("Part B complete")
    if args.stage in {"report", "evaluate", "all"}:
        build_report(cfg)
        print(f"Report complete: {Path(cfg['output']['directory']) / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
