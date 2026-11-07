from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load YAML and resolve all configured paths relative to the config file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    root = config_path.parent
    for key in ("accepted_path", "rejected_path", "prepared_path"):
        value = Path(cfg["data"][key]).expanduser()
        cfg["data"][key] = str(value if value.is_absolute() else (root / value).resolve())
    output = Path(cfg["output"]["directory"]).expanduser()
    cfg["output"]["directory"] = str(output if output.is_absolute() else (root / output).resolve())
    validate_config(cfg)
    return cfg, root


def validate_config(cfg: dict[str, Any]) -> None:
    alphas = [float(x) for x in cfg["evaluation"]["alphas"]]
    gammas = [float(x) for x in cfg["evaluation"]["gamma_grid"]]
    if any(not 0.0 < x < 1.0 for x in alphas):
        raise ValueError("Every alpha must be strictly between zero and one.")
    if any(x < 1.0 for x in gammas) or gammas != sorted(set(gammas)):
        raise ValueError("gamma_grid must be unique, increasing, and at least one.")
    if gammas[0] != 1.0:
        raise ValueError("gamma_grid must begin at 1.0 for breakdown reporting.")
    gamma0 = float(cfg["part_b"]["gamma0"])
    if gamma0 not in gammas:
        raise ValueError("part_b.gamma0 must be explicitly predeclared in evaluation.gamma_grid.")
    for name in ("p1_clip", "propensity_clip"):
        lo, hi = map(float, cfg["evaluation"][name])
        if not 0.0 < lo < hi < 1.0:
            raise ValueError(f"{name} must be a strict subinterval of (0,1).")
    if int(cfg["evaluation"]["n_folds"]) != 5:
        raise ValueError("This protocol predeclares five-fold cross-fitting; set n_folds: 5.")
    if not 0.0 < float(cfg["part_b"]["batch_fraction_hidden"]) <= 1.0:
        raise ValueError("batch_fraction_hidden must be in (0,1].")


def feature_columns(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    data_cfg = cfg["data"]["features"]
    return list(data_cfg["numeric"]), list(data_cfg["categorical"])
