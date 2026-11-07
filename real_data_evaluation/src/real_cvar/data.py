from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ACCEPTED_ALIASES = {
    "requested_amount": ("loan_amnt", "funded_amnt"),
    "dti": ("dti",),
    "risk_score_low": ("fico_range_low",),
    "risk_score_high": ("fico_range_high",),
    "risk_score": ("fico", "risk_score"),
    "employment_length": ("emp_length",),
    "state": ("addr_state",),
    "zip3": ("zip_code",),
    "purpose": ("purpose", "title"),
    "policy_code": ("policy_code",),
    "date": ("issue_d",),
    "loan_status": ("loan_status",),
}

REJECTED_ALIASES = {
    "requested_amount": ("Amount Requested", "amount_requested"),
    "dti": ("Debt-To-Income Ratio", "debt_to_income_ratio"),
    "risk_score": ("Risk_Score", "risk_score"),
    "employment_length": ("Employment Length", "employment_length"),
    "state": ("State", "state"),
    "zip3": ("Zip Code", "zip_code"),
    "purpose": ("Loan Title", "loan_title"),
    "policy_code": ("Policy Code", "policy_code"),
    "date": ("Application Date", "application_date"),
}

DEFAULT_STATUSES = {
    "charged off",
    "default",
    "does not meet the credit policy. status:charged off",
}
NONDEFAULT_STATUSES = {
    "fully paid",
    "does not meet the credit policy. status:fully paid",
}


def _normalized_lookup(columns: Iterable[str]) -> dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in columns}


def _resolve(columns: Iterable[str], aliases: dict[str, tuple[str, ...]], required: set[str]) -> dict[str, str]:
    lookup = _normalized_lookup(columns)
    result: dict[str, str] = {}
    for canonical, choices in aliases.items():
        for candidate in choices:
            if candidate.strip().lower() in lookup:
                result[canonical] = lookup[candidate.strip().lower()]
                break
        if canonical in required and canonical not in result:
            raise ValueError(f"Required field {canonical!r} was not found. Tried aliases {choices}.")
    return result


def inspect_schema(path: str | Path, accepted: bool) -> dict[str, str]:
    columns = pd.read_csv(path, nrows=0, low_memory=False).columns
    aliases = ACCEPTED_ALIASES if accepted else REJECTED_ALIASES
    required = {"requested_amount", "dti", "employment_length", "state", "zip3", "purpose", "date"}
    if accepted:
        required.add("loan_status")
    mapping = _resolve(columns, aliases, required)
    if accepted and not ({"risk_score"} <= mapping.keys() or {"risk_score_low", "risk_score_high"} <= mapping.keys()):
        raise ValueError("Accepted file needs fico_range_low/high or a risk-score column.")
    if not accepted and "risk_score" not in mapping:
        raise ValueError("Rejected file needs Risk_Score; it is a key common pre-decision feature.")
    return mapping


def _employment_years(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.lower().str.strip()
    # Force ordinary float64 before assigning 0.5 for "< 1 year".  Pandas 2.2
    # otherwise preserves the extracted digits as nullable Int64 and rejects the
    # fractional replacement.
    values = pd.to_numeric(
        text.str.extract(r"(\d+)", expand=False), errors="coerce"
    ).astype("float64")
    values = values.mask(text.str.contains("<", na=False), 0.5)
    values = values.mask(text.str.contains(r"10\+", regex=True, na=False), 10.0)
    return values.clip(0.0, 10.0)


def _zip3(series: pd.Series) -> pd.Series:
    return series.astype("string").str.extract(r"(\d{3})", expand=False).fillna("UNK")


def _purpose(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.lower().fillna("")
    direct = text.str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    rules = [
        (r"debt|consolid", "debt_consolidation"),
        (r"credit[_ ]?card", "credit_card"),
        (r"home|house|improv", "home_improvement"),
        (r"small[_ ]?business|business", "small_business"),
        (r"major[_ ]?purchase|purchase", "major_purchase"),
        (r"medical", "medical"),
        (r"moving", "moving"),
        (r"vacation|travel", "vacation"),
        (r"wedding", "wedding"),
        (r"car|auto", "car"),
        (r"education|school|college", "education"),
        (r"renewable|green", "renewable_energy"),
    ]
    result = pd.Series("other", index=series.index, dtype="string")
    for pattern, label in rules:
        result = result.mask(text.str.contains(pattern, regex=True, na=False), label)
    known = {
        "debt_consolidation", "credit_card", "home_improvement", "small_business",
        "major_purchase", "medical", "moving", "vacation", "wedding", "car",
        "education", "renewable_energy", "house", "other",
    }
    result = result.mask(direct.isin(known), direct)
    return result


def _numeric(series: pd.Series, percent: bool = False) -> pd.Series:
    text = series.astype("string").str.replace(",", "", regex=False)
    if percent:
        text = text.str.replace("%", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _standardize_chunk(chunk: pd.DataFrame, mapping: dict[str, str], accepted: bool, year: int) -> pd.DataFrame:
    date = pd.to_datetime(chunk[mapping["date"]], errors="coerce", format="mixed")
    keep = date.dt.year.eq(year)
    if not keep.any():
        return pd.DataFrame()
    cur = chunk.loc[keep]
    out = pd.DataFrame(index=cur.index)
    out["requested_amount"] = _numeric(cur[mapping["requested_amount"]])
    out["dti"] = _numeric(cur[mapping["dti"]], percent=True)
    if accepted and "risk_score_low" in mapping:
        low = _numeric(cur[mapping["risk_score_low"]])
        high = _numeric(cur[mapping["risk_score_high"]])
        out["risk_score"] = (low + high) / 2.0
    else:
        out["risk_score"] = _numeric(cur[mapping["risk_score"]])
    # Public rejected files can encode unavailable bureau scores as zero.  Treat
    # values outside the FICO scale as missing instead of leaking a file-specific
    # sentinel into the acceptance model.
    out["risk_score"] = out["risk_score"].where(out["risk_score"].between(300.0, 850.0))
    out["dti"] = out["dti"].where(out["dti"] >= 0.0)
    out["requested_amount"] = out["requested_amount"].where(out["requested_amount"] > 0.0)
    out["employment_years"] = _employment_years(cur[mapping["employment_length"]])
    out["state"] = cur[mapping["state"]].astype("string").str.upper().str.strip().fillna("UNK")
    out["zip3"] = _zip3(cur[mapping["zip3"]])
    out["purpose"] = _purpose(cur[mapping["purpose"]])
    if "policy_code" in mapping:
        out["policy_code"] = cur[mapping["policy_code"]].astype("string").str.strip().fillna("UNK")
    else:
        out["policy_code"] = "UNK"
    out["application_month"] = date.loc[keep].dt.month.astype("Int64").to_numpy()
    out["S"] = np.int8(1 if accepted else 0)
    if accepted:
        status = cur[mapping["loan_status"]].astype("string").str.lower().str.strip()
        outcome = pd.Series(np.nan, index=cur.index, dtype=float)
        outcome.loc[status.isin(DEFAULT_STATUSES)] = 1.0
        outcome.loc[status.isin(NONDEFAULT_STATUSES)] = 0.0
        out["Y"] = outcome
        out = out.loc[out["Y"].notna()]
    else:
        out["Y"] = np.nan
    return out.reset_index(drop=True)


def _read_standardized(path: str | Path, mapping: dict[str, str], accepted: bool, year: int, chunksize: int) -> pd.DataFrame:
    usecols = sorted(set(mapping.values()))
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        standardized = _standardize_chunk(chunk, mapping, accepted, year)
        if not standardized.empty:
            frames.append(standardized)
    if not frames:
        raise ValueError(f"No eligible {year} rows found in {path}.")
    return pd.concat(frames, ignore_index=True)


def add_subgroups(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["subgroup_fico"] = pd.cut(
        data["risk_score"], [-np.inf, 659.5, 699.5, 739.5, np.inf],
        labels=["fico_<660", "fico_660_699", "fico_700_739", "fico_740_plus"],
    ).astype("string").fillna("fico_missing")
    data["subgroup_dti"] = pd.cut(
        data["dti"], [-np.inf, 10.0, 20.0, 30.0, np.inf],
        labels=["dti_0_10", "dti_10_20", "dti_20_30", "dti_30_plus"],
    ).astype("string").fillna("dti_missing")
    amount_rank = data["requested_amount"].rank(method="average", pct=True)
    data["subgroup_amount"] = pd.cut(
        amount_rank, [0.0, 0.25, 0.50, 0.75, 1.0], include_lowest=True,
        labels=["amount_q1", "amount_q2", "amount_q3", "amount_q4"],
    ).astype("string").fillna("amount_missing")
    return data


def prepare_lendingclub(cfg: dict[str, Any]) -> pd.DataFrame:
    accepted_path = Path(cfg["data"]["accepted_path"])
    rejected_path = Path(cfg["data"]["rejected_path"])
    for path in (accepted_path, rejected_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing LendingClub input: {path}")
    year = int(cfg["cohort_year"])
    chunksize = int(cfg["data"]["chunksize"])
    accepted_map = inspect_schema(accepted_path, accepted=True)
    rejected_map = inspect_schema(rejected_path, accepted=False)
    accepted = _read_standardized(accepted_path, accepted_map, True, year, chunksize)
    rejected = _read_standardized(rejected_path, rejected_map, False, year, chunksize)
    accepted["source_row_id"] = "A" + accepted.index.astype(str)
    rejected["source_row_id"] = "R" + rejected.index.astype(str)
    combined = add_subgroups(pd.concat([accepted, rejected], ignore_index=True))
    expected = cfg.get("expected_counts", {})
    actual = {"accepted": len(accepted), "rejected": len(rejected)}
    for name, count in actual.items():
        if name in expected and int(expected[name]) != count:
            warnings.warn(
                f"{name} count is {count:,}, not documented reference count {int(expected[name]):,}. "
                "LendingClub snapshots differ; the run will report the observed counts.",
                stacklevel=2,
            )
    output = Path(cfg["data"]["prepared_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    metadata_dir = Path(cfg["output"]["directory"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    features = ["requested_amount", "dti", "risk_score", "employment_years", "state", "zip3", "purpose"]
    metadata = {
        "cohort_year": year,
        "accepted_source_name": accepted_path.name,
        "rejected_source_name": rejected_path.name,
        "accepted_schema_mapping": accepted_map,
        "rejected_schema_mapping": rejected_map,
        "cohort_date_caveat": "accepted rows filtered by issue date; rejected rows filtered by application date",
        "calendar_month_excluded_from_model_features": True,
        "accepted_mature_count": len(accepted),
        "rejected_count": len(rejected),
        "total_count": len(combined),
        "default_count_among_accepted": int(accepted["Y"].sum()),
        "feature_missing_fraction": {name: float(combined[name].isna().mean()) for name in features},
        "policy_code_stored_but_excluded_from_model_features": True,
    }
    with (metadata_dir / "preparation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return combined


def load_prepared(cfg: dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["data"]["prepared_path"])
    if not path.exists():
        raise FileNotFoundError(f"Prepared cohort not found: {path}. Run the prepare stage first.")
    return pd.read_parquet(path)
