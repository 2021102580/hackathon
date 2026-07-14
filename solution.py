"""Self-contained staged income prediction pipeline.

Only the labelled training table and the unlabelled inference table are inputs.
Every structural, out-of-fold and meta prediction is rebuilt in this process.
Generated CSV files are audit outputs only: downstream stages consume in-memory
arrays and never read prediction artifacts. ``dt`` is an as-of date used only
for relative ages and optional strict latest-date evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor

ID_COL, TARGET_COL, WEIGHT_COL, ASOF_COL = "id", "target", "w", "dt"
STAGES = ["center", "tail150", "tail200", "tail_mix", "catboost_mix", "structural_sources", "structural_final", "anchor_v2"]
CENTER = 84017.079961153
TAIL_THRESHOLDS = (150000.0, 200000.0)
KNOWN_CATEGORICAL = {"gender", "adminarea", "addrref", "city_smart_name", "dp_address_unique_regions", "dp_ewb_last_organization"}
SOURCE_COLUMNS = {
    "salary": ["salary_6to12m_avg"],
    "payout": ["dp_payoutincomedata_payout_avg_6_month", "dp_payoutincomedata_payout_avg_3_month", "dp_payoutincomedata_payout_avg_prev_year"],
    "ils": ["dp_ils_avg_salary_1y", "dp_ils_avg_salary_2y", "dp_ils_avg_salary_3y"],
}
META_RAW = [
    "salary_6to12m_avg", "incomeValue", "dp_ils_avg_salary_1y", "dp_ils_avg_salary_2y", "dp_ils_avg_salary_3y",
    "dp_payoutincomedata_payout_avg_6_month", "dp_payoutincomedata_payout_avg_3_month",
    "dp_payoutincomedata_payout_avg_prev_year", "salary_median_in_gex_r1", "per_capita_income_rur_amt",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",", low_memory=False)


def to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float32")
    return pd.to_numeric(s.astype("string").str.replace(",", ".", regex=False), errors="coerce").astype("float32")


def parse_iso_date(s: pd.Series, name: str, require_valid: bool = True) -> pd.Series:
    text = s.astype("string")
    nonnull = text.notna() & text.str.strip().ne("")
    iso = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    if bool((nonnull & ~iso).any()):
        raise ValueError(f"{name} must use ISO YYYY-MM-DD; got {text[nonnull & ~iso].head(3).tolist()}")
    parsed = pd.to_datetime(text.where(iso), format="%Y-%m-%d", errors="coerce")
    if require_valid and bool((nonnull & parsed.isna()).any()):
        raise ValueError(f"{name} contains invalid dates")
    return parsed


def weighted_mae(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float | None:
    return None if float(w.sum()) <= 0 else float(np.average(np.abs(y - p), weights=w))


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, sep=";", index=False)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def stable_hash(ids: pd.Series, seed: int) -> np.ndarray:
    """Stable, order-independent and genuinely seed-sensitive SplitMix64 hash."""
    x = pd.util.hash_pandas_object(ids.astype("string"), index=False).to_numpy("uint64")
    seed_mix = np.uint64((int(seed) * 0x9E3779B185EBCA87 + 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
    x = x + seed_mix
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def canonicalize(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    return frame.iloc[np.argsort(stable_hash(frame[ID_COL], seed), kind="stable")].reset_index(drop=True)


def make_folds(frame: pd.DataFrame, count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    k = max(2, min(count, len(frame)))
    order = np.argsort(stable_hash(frame[ID_COL], seed), kind="stable")
    assignment = np.empty(len(frame), dtype="int16")
    assignment[order] = np.arange(len(frame)) % k
    return [(np.flatnonzero(assignment != f), np.flatnonzero(assignment == f)) for f in range(k)]


def domain(name: str) -> str:
    c = name.lower()
    if any(x in c for x in ("salary", "income", "payout")): return "income"
    if any(x in c for x in ("bki", "loan", "pil", "overdue", "micro", "outstand", "max_limit")): return "credit"
    if any(x in c for x in ("turn_", "amount__", "cashflow", "transaction", "sbp", "perevod", "platezh")): return "transactions"
    if any(x in c for x in ("dp_ils", "employment", "employ", "seniority", "organization")): return "employment"
    if any(x in c for x in ("adminarea", "addrref", "city", "region", "geo", "gender")): return "demography"
    return "other"


class FeatureBuilder:
    """All state is learned from the supplied fit table, never inference rows."""
    def __init__(self) -> None:
        self.used: list[str] = []
        self.required: list[str] = []
        self.categories: dict[str, list[str]] = {}
        self.domains: dict[str, list[str]] = {}
        self.thresholds: dict[str, float] = {}
        self.output: list[str] = []

    @staticmethod
    def is_category(name: str, s: pd.Series) -> bool:
        if name in KNOWN_CATEGORICAL or isinstance(s.dtype, pd.CategoricalDtype): return True
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            observed = int(s.notna().sum())
            return observed == 0 or int(to_numeric(s).notna().sum()) / observed < .90
        return False

    def fit(self, raw: pd.DataFrame) -> "FeatureBuilder":
        self.__init__()
        for c in raw.columns:
            if c in {ID_COL, TARGET_COL, WEIGHT_COL, ASOF_COL}: continue
            if c == "period_last_act_ad":
                if raw[c].notna().any(): self.used.append(c)
                continue
            if raw[c].nunique(dropna=False) <= 1: continue
            self.used.append(c)
            if self.is_category(c, raw[c]):
                vals = sorted(raw[c].astype("string").fillna("__MISSING__").unique().tolist())
                for special in ("__MISSING__", "__OTHER__"):
                    if special not in vals: vals.append(special)
                self.categories[c] = vals
        self.required = list(self.used)
        if "period_last_act_ad" in self.used: self.required.append(ASOF_COL)
        for c in self.used:
            if c not in self.categories and c != "period_last_act_ad": self.domains.setdefault(domain(c), []).append(c)
        for d, cols in self.domains.items():
            self.thresholds[d] = float(np.quantile(raw[cols].notna().sum(axis=1), .65))
        self.output = self._transform(raw).columns.tolist()
        return self

    def _transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.required if c not in raw]
        if missing: raise ValueError(f"Input is missing fitted feature columns: {missing[:10]}")
        data: dict[str, pd.Series] = {}
        numeric: list[str] = []
        asof = parse_iso_date(raw[ASOF_COL], ASOF_COL) if "period_last_act_ad" in self.used else None
        for c in self.used:
            if c in self.categories:
                s = raw[c].astype("string").fillna("__MISSING__")
                s = s.where(s.isin(set(self.categories[c])), "__OTHER__")
                data[c] = s.astype(pd.CategoricalDtype(self.categories[c]))
            elif c == "period_last_act_ad":
                event = parse_iso_date(raw[c], c, require_valid=False)
                age = ((asof.dt.year - event.dt.year) * 12 + asof.dt.month - event.dt.month).astype("float32")
                age[(event.dt.year < 1900) | (age < 0)] = np.nan
                data["period_last_act_ad_age_months"] = age; numeric.append("period_last_act_ad_age_months")
            else:
                data[c] = to_numeric(raw[c]); numeric.append(c)
        out = pd.DataFrame(data, index=raw.index)
        out["row_missing_count"] = out[numeric].isna().sum(axis=1).astype("float32") if numeric else 0.
        for d, cols in self.domains.items():
            cnt = raw[cols].notna().sum(axis=1).astype("float32")
            out[f"availability_{d}_count"] = cnt
            out[f"availability_{d}_rate"] = (cnt / len(cols)).astype("float32")
            out[f"availability_{d}_rich"] = (cnt >= self.thresholds[d]).astype("float32")
        # Missing is informative, but zero is also the natural value for absent transaction/credit aggregates.
        for c in out.columns:
            if str(out[c].dtype) == "category": continue
            lc = c.lower()
            if any(x in lc for x in ("turn_", "amount__", "cashflow", "transaction", "sbp", "bki", "loan", "overdue", "outstand", "max_limit")):
                out[c] = out[c].fillna(0).astype("float32")
        return out

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not self.output: raise RuntimeError("FeatureBuilder is not fitted")
        return self._transform(raw).reindex(columns=self.output)

    def fit_transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw).transform(raw)


@dataclass(frozen=True)
class Config:
    folds: int = 3
    seed: int = 2026
    device: str = "gpu"
    center_clf: int = 320
    center_reg: int = 540
    tail_clf: int = 170
    tail_reg: int = 230
    source_reg: int = 600
    low_clf: int = 260
    low_reg: int = 500
    catboost_iterations: int = 700
    meta_trees: int = 180
    signed_trees: int = 300
    selected_trees: int = 300
    selector_trees: int = 260
    top_k: int = 100
    endpoint_alpha: float = .75
    use_catboost: bool = True

    @classmethod
    def fast(cls, folds: int, seed: int, device: str) -> "Config":
        return cls(folds=folds, seed=seed, device=device, center_clf=10, center_reg=12, tail_clf=9, tail_reg=10,
                   source_reg=10, low_clf=8, low_reg=10, catboost_iterations=0, meta_trees=12,
                   signed_trees=14, selected_trees=12, selector_trees=12, top_k=20, use_catboost=False)


@dataclass
class Runtime:
    requested_device: str
    lgb_cpu_fallbacks: int = 0
    xgb_cpu_fallbacks: int = 0
    catboost_cpu_fallbacks: int = 0


def lgb_params(trees: int, seed: int, cfg: Config, objective: str = "mae", leaves: int = 63, min_child: int = 50) -> dict:
    return {"objective": objective, "n_estimators": trees, "learning_rate": .03, "num_leaves": leaves,
            "min_child_samples": min_child, "colsample_bytree": .8, "reg_alpha": .1, "reg_lambda": 3.,
            "random_state": seed, "n_jobs": -1, "verbosity": -1, "device_type": cfg.device,
            "max_bin": 255 if cfg.device == "gpu" else 511, "deterministic": True,
            "force_col_wise": cfg.device == "cpu"}


def fit_lgb(kind: str, params: dict, x: pd.DataFrame, y: np.ndarray, w: np.ndarray, cats: list[str], runtime: Runtime):
    cls = lgb.LGBMClassifier if kind == "classifier" else lgb.LGBMRegressor
    support = np.isfinite(w) & (w > 0)
    if int(support.sum()) < 2: raise ValueError("Model subset has fewer than two positive-weight rows")
    x, y, w = x.loc[support], np.asarray(y)[support], np.asarray(w)[support]
    try:
        m = cls(**params); m.fit(x, y, sample_weight=w, categorical_feature=cats); return m
    except lgb.basic.LightGBMError:
        if params.get("device_type") != "gpu": raise
        runtime.lgb_cpu_fallbacks += 1
        p = dict(params); p.update(device_type="cpu", max_bin=511, force_col_wise=True)
        m = cls(**p); m.fit(x, y, sample_weight=w, categorical_feature=cats); return m


def clip_factory(y: np.ndarray):
    lo, hi = float(np.min(y)), float(np.max(y))
    return lambda p: np.clip(np.asarray(p, float), lo, hi)


def source_anchor(raw: pd.DataFrame, name: str) -> np.ndarray:
    out = pd.Series(np.nan, index=raw.index, dtype="float64")
    for c in SOURCE_COLUMNS[name]:
        if c in raw: out = out.fillna(to_numeric(raw[c]).astype("float64"))
    return out.to_numpy(float)


def source_known_masks(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: np.isfinite(source_anchor(raw, name)) & (source_anchor(raw, name) > 0) for name in ("salary", "payout", "ils")}


def source_masks(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    known = source_known_masks(raw)
    s = known["salary"]
    p = known["payout"] & ~s
    i = known["ils"] & ~s & ~p
    return {"salary": s, "payout": p, "ils": i}


def safe_binary_probability(xtr, xpr, label, w, cats, trees, seed, cfg, runtime) -> np.ndarray:
    support = np.isfinite(w) & (w > 0)
    supported_labels = np.asarray(label)[support]
    if len(supported_labels) == 0: return np.zeros(len(xpr), dtype=float)
    classes = np.unique(supported_labels)
    if classes.size < 2: return np.full(len(xpr), float(classes[0]))
    p = lgb_params(trees, seed, cfg, "binary", 47, 70)
    m = fit_lgb("classifier", p, xtr, np.asarray(label).astype("int8"), w, cats, runtime)
    return m.predict_proba(xpr)[:, 1]


def fit_structural(fit_raw: pd.DataFrame, pred_raw: pd.DataFrame, cfg: Config, runtime: Runtime) -> dict[str, np.ndarray]:
    builder = FeatureBuilder(); xtr = builder.fit_transform(fit_raw); xpr = builder.transform(pred_raw)
    cats = [c for c in xtr if str(xtr[c].dtype) == "category"]
    y = to_numeric(fit_raw[TARGET_COL]).to_numpy(float); w = to_numeric(fit_raw[WEIGHT_COL]).to_numpy(float) if WEIGHT_COL in fit_raw else np.ones(len(fit_raw))
    positive_weight = np.isfinite(w) & (w > 0)
    if int(positive_weight.sum()) < 2:
        fallback = float(y[positive_weight][0]) if positive_weight.any() else float(np.clip(CENTER, np.min(y), np.max(y)))
        return {name: np.full(len(pred_raw), fallback, dtype=float) for name in STAGES}
    clip = clip_factory(y[positive_weight])
    group = (y >= CENTER).astype("int8")
    prob = safe_binary_probability(xtr, xpr, group, w, cats, cfg.center_clf, cfg.seed + 11, cfg, runtime)
    experts = []
    for g in (0, 1):
        mask = (group == g) & positive_weight
        if mask.sum() < 2: experts.append(np.full(len(xpr), np.average(y[positive_weight], weights=w[positive_weight]))); continue
        m = fit_lgb("regressor", lgb_params(cfg.center_reg, cfg.seed + 20 + g, cfg, "mae", 63, 55), xtr.loc[mask], y[mask], w[mask], cats, runtime)
        experts.append(clip(m.predict(xpr)))
    sharp = np.column_stack([1 - prob, prob]) ** 2; sharp /= np.maximum(sharp.sum(axis=1, keepdims=True), 1e-12)
    center = clip((np.column_stack(experts) * sharp).sum(axis=1))
    tails = []
    for j, (thr, gamma, min_base) in enumerate(((150000., 3., 100000.), (200000., 2., 150000.))):
        label = (y >= thr).astype("int8")
        tp = safe_binary_probability(xtr, xpr, label, w, cats, cfg.tail_clf, cfg.seed + 40 + j, cfg, runtime)
        mask = (label == 1) & positive_weight
        if mask.sum() < 2:
            tails.append(center.copy()); continue
        m = fit_lgb("regressor", lgb_params(cfg.tail_reg, cfg.seed + 50 + j, cfg, "regression_l2", 47, 35), xtr.loc[mask], np.log1p(y[mask]), w[mask], cats, runtime)
        high = clip(np.expm1(m.predict(xpr)))
        tails.append(clip(center + .5 * tp ** gamma * (center >= min_base) * np.maximum(high - center, 0)))
    tail150, tail200 = tails; tail_mix = .5 * tail150 + .5 * tail200
    cat_pred = tail_mix.copy()
    if cfg.use_catboost and cfg.catboost_iterations > 0:
        a = xtr.copy(); b = xpr.copy(); cat_idx = []
        for idx, c in enumerate(a.columns):
            if str(a[c].dtype) == "category":
                a[c] = a[c].astype("string").fillna("__MISSING__").astype(str); b[c] = b[c].astype("string").fillna("__MISSING__").astype(str); cat_idx.append(idx)
            else:
                a[c] = pd.to_numeric(a[c], errors="coerce").astype("float32"); b[c] = pd.to_numeric(b[c], errors="coerce").astype("float32")
        task = "GPU" if cfg.device == "gpu" else "CPU"
        kwargs = dict(iterations=cfg.catboost_iterations, learning_rate=.05, depth=8, loss_function="MAE", l2_leaf_reg=5,
                      random_seed=cfg.seed + 70, task_type=task, verbose=False, allow_writing_files=False)
        train_pool = Pool(a.loc[positive_weight], np.log1p(y[positive_weight]), weight=w[positive_weight], cat_features=cat_idx)
        try:
            model = CatBoostRegressor(**kwargs); model.fit(train_pool); cb = clip(np.expm1(model.predict(Pool(b, cat_features=cat_idx))))
        except Exception:
            if task != "GPU": raise
            runtime.catboost_cpu_fallbacks += 1; kwargs["task_type"] = "CPU"
            model = CatBoostRegressor(**kwargs); model.fit(train_pool); cb = clip(np.expm1(model.predict(Pool(b, cat_features=cat_idx))))
        cat_pred = clip(.84 * tail_mix + .16 * cb)
    # Historical raw source specialists, retained as a diversity stage.
    structural_sources = cat_pred.copy(); used = np.zeros(len(pred_raw), bool)
    train_masks, pred_masks = source_masks(fit_raw), source_masks(pred_raw)
    train_known = source_known_masks(fit_raw)
    for j, (name, alpha) in enumerate((("salary", .65), ("payout", .25), ("ils", .30))):
        # Diversity source experts learn from every positive-weight known row;
        # deployment remains exclusive salary -> payout -> ILS.
        mt, mp = train_known[name] & positive_weight, pred_masks[name] & ~used
        if mt.sum() >= 2 and mp.any():
            m = fit_lgb("regressor", lgb_params(cfg.source_reg, cfg.seed + 90 + j, cfg, "mae", 63, 45), xtr.loc[mt], y[mt], w[mt], cats, runtime)
            structural_sources[mp] = (1 - alpha) * structural_sources[mp] + alpha * clip(m.predict(xpr.loc[mp])); used |= mp
    # Low-side soft correction.
    low_label = (y <= 50000.).astype("int8")
    low_prob = safe_binary_probability(xtr, xpr, low_label, w, cats, cfg.low_clf, cfg.seed + 110, cfg, runtime)
    low_mask = (low_label == 1) & positive_weight
    low_delta = np.zeros(len(xpr))
    if low_mask.sum() >= 2:
        m = fit_lgb("regressor", lgb_params(cfg.low_reg, cfg.seed + 111, cfg, "mae", 47, 40), xtr.loc[low_mask], y[low_mask], w[low_mask], cats, runtime)
        low_pred = np.clip(m.predict(xpr), max(0., float(y.min())), 50000.)
        low_delta = -.75 * low_prob ** 3 * (center <= 150000.) * np.maximum(center - low_pred, 0)
    structural_final = clip(structural_sources + low_delta)
    # Hierarchy-matched explicit anchor residual specialists.
    anchor_v2 = cat_pred.copy(); used = np.zeros(len(pred_raw), bool)
    for j, (name, variant, alpha) in enumerate((("salary", "log", .70), ("payout", "residual", .30), ("ils", "residual", .35))):
        atr, apr = source_anchor(fit_raw, name), source_anchor(pred_raw, name)
        # Anchor-v2 uses the identical hierarchy in fit and deployment.
        mt, mp = train_masks[name] & positive_weight, pred_masks[name] & ~used
        if mt.sum() >= 2 and mp.any():
            target = np.log(np.maximum(y[mt], 1) / np.maximum(atr[mt], 1)) if variant == "log" else y[mt] - atr[mt]
            m = fit_lgb("regressor", lgb_params(cfg.source_reg, cfg.seed + 130 + j, cfg, "mae", 63, 45), xtr.loc[mt], target, w[mt], cats, runtime)
            q = m.predict(xpr.loc[mp]); expert = apr[mp] * np.exp(np.clip(q, -4, 4)) if variant == "log" else apr[mp] + q
            anchor_v2[mp] = (1 - alpha) * anchor_v2[mp] + alpha * clip(expert); used |= mp
    anchor_v2 = clip(anchor_v2 + low_delta)
    return {"center": center, "tail150": tail150, "tail200": tail200, "tail_mix": tail_mix,
            "catboost_mix": cat_pred, "structural_sources": structural_sources,
            "structural_final": structural_final, "anchor_v2": anchor_v2}


def numeric_frame(x: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = x[cols].copy()
    for c in cols:
        if str(z[c].dtype) == "category": z[c] = z[c].cat.codes.replace(-1, np.nan)
    return z.astype("float32")


def signed_encode(y: np.ndarray, base: np.ndarray) -> np.ndarray:
    d = y - base; return np.sign(d) * np.log1p(np.abs(d) / np.maximum(base, 1))


def signed_decode(base: np.ndarray, q: np.ndarray, shrink: float) -> np.ndarray:
    z = shrink * q; return base * (1 + np.sign(z) * np.expm1(np.abs(z)))


def build_meta(raw: pd.DataFrame, x: pd.DataFrame, stages: dict[str, np.ndarray]) -> pd.DataFrame:
    z = x.copy(); base = stages["anchor_v2"]
    for name in STAGES: z[f"stage_{name}"] = np.asarray(stages[name], dtype="float32")
    z["stage_log_anchor"] = np.log1p(base).astype("float32")
    z["stage_tail_lift"] = (stages["tail_mix"] - stages["center"]).astype("float32")
    z["stage_catboost_delta"] = (stages["catboost_mix"] - stages["tail_mix"]).astype("float32")
    z["stage_source_delta"] = (stages["structural_sources"] - stages["catboost_mix"]).astype("float32")
    vals = []
    for c in META_RAW:
        if c not in z: continue
        v = pd.to_numeric(z[c], errors="coerce").to_numpy(float)
        z[f"anchor_lr_{c}"] = np.clip(np.log1p(np.maximum(v, 0)) - np.log1p(np.maximum(base, 0)), -5, 5).astype("float32")
        vals.append(v)
    if vals:
        arr = np.column_stack(vals); z["anchor_known_count"] = np.isfinite(arr).sum(1).astype("float32")
    return z


def xgb_model(trees: int, seed: int, cfg: Config, depth: int = 5) -> XGBRegressor:
    return XGBRegressor(n_estimators=trees, learning_rate=.025, max_depth=depth, min_child_weight=20,
                        subsample=.85, colsample_bytree=.78, reg_lambda=12, reg_alpha=.2,
                        objective="reg:absoluteerror", tree_method="hist", device="cuda" if cfg.device == "gpu" else "cpu",
                        random_state=seed, n_jobs=-1)


def fit_xgb(model: XGBRegressor, x: pd.DataFrame, y: np.ndarray, w: np.ndarray, runtime: Runtime) -> XGBRegressor:
    support = np.isfinite(w) & (w > 0)
    if int(support.sum()) < 2: raise ValueError("XGBoost subset has fewer than two positive-weight rows")
    x, y, w = x.loc[support], np.asarray(y)[support], np.asarray(w)[support]
    try:
        model.fit(x, y, sample_weight=w); return model
    except Exception:
        if model.get_params().get("device") != "cuda": raise
        runtime.xgb_cpu_fallbacks += 1
        model.set_params(device="cpu"); model.fit(x, y, sample_weight=w); return model


def run_graph(train: pd.DataFrame, test: pd.DataFrame, cfg: Config, runtime: Runtime) -> tuple[np.ndarray, dict, dict[str, pd.DataFrame]]:
    if cfg.folds < 2: raise ValueError("folds must be at least 2")
    raw_order_by_id = pd.Series(np.arange(len(train), dtype=int), index=train[ID_COL])
    train = canonicalize(train, cfg.seed)
    raw_order = raw_order_by_id.loc[train[ID_COL]].to_numpy(int)
    y = to_numeric(train[TARGET_COL]).to_numpy(float); w = to_numeric(train[WEIGHT_COL]).to_numpy(float) if WEIGHT_COL in train else np.ones(len(train))
    clip = clip_factory(y); folds = make_folds(train, cfg.folds, cfg.seed)
    oof = {s: np.empty(len(train), float) for s in STAGES}
    for f, (tr, va) in enumerate(folds):
        log(f"Structural OOF fold {f + 1}/{len(folds)}")
        local_cfg = Config(**{**asdict(cfg), "seed": cfg.seed + 1000 * f})
        pred = fit_structural(train.iloc[tr], train.iloc[va].drop(columns=[TARGET_COL, WEIGHT_COL], errors="ignore"), local_cfg, runtime)
        for s in STAGES: oof[s][va] = pred[s]
    log("Structural full-train inference stages")
    test_stages = fit_structural(train, test, cfg, runtime)
    # Common train-fitted feature space for meta learners.
    builder = FeatureBuilder(); x = builder.fit_transform(train); xt = builder.transform(test)
    mo = build_meta(train, x, oof); mt = build_meta(test, xt, test_stages)
    base_oof, base_test = oof["anchor_v2"], test_stages["anchor_v2"]
    pred_cols = [c for c in mo if c.startswith("stage_")]
    block_cols = [c for c in mo if c.startswith("availability_")]
    ratio_cols = [c for c in mo if c.startswith("anchor_lr_") and any(p in c for p in ("salary_6to12m", "dp_payout", "dp_ils"))]
    meta_cols = list(dict.fromkeys(pred_cols + [c for c in META_RAW if c in mo] + block_cols + ["row_missing_count"] + ratio_cols))
    meta_cats = [c for c in meta_cols if str(mo[c].dtype) == "category"]
    # Residual-v3 is the common fallback/anchor for every endpoint.
    log_ratio_target = np.log(np.maximum(y, 1) / np.maximum(base_oof, 1))
    p = lgb_params(cfg.meta_trees, cfg.seed + 3001, cfg, "mae", 63, 100); p["learning_rate"] = .02
    residual_model = fit_lgb("regressor", p, mo[meta_cols], log_ratio_target, w, meta_cats, runtime)
    v3_train = clip(base_oof * np.exp(np.clip(.575 * residual_model.predict(mo[meta_cols]), -2, 2)))
    v3 = clip(base_test * np.exp(np.clip(.575 * residual_model.predict(mt[meta_cols]), -2, 2)))
    signed_target = signed_encode(y, base_oof)
    zm, zmt = numeric_frame(mo, meta_cols), numeric_frame(mt, meta_cols)

    # Historical direct log(target/base) endpoint, separate from signed residual.
    direct_train_parts, direct_parts = [], []
    for seed in (cfg.seed + 3051, cfg.seed + 3052, cfg.seed + 3053):
        m = fit_xgb(xgb_model(cfg.signed_trees, seed, cfg, 5), zm, log_ratio_target, w, runtime)
        direct_train_parts.append(clip(base_oof * np.exp(np.clip(.9 * m.predict(zm), -2, 2))))
        direct_parts.append(clip(base_test * np.exp(np.clip(.9 * m.predict(zmt), -2, 2))))
    direct_raw_train, direct_raw = np.mean(direct_train_parts, axis=0), np.mean(direct_parts, axis=0)

    signed_train_parts, signed_parts = [], []
    for seed in (cfg.seed + 3101, cfg.seed + 3102, cfg.seed + 3103):
        m = fit_xgb(xgb_model(cfg.signed_trees, seed, cfg, 5), zm, signed_target, w, runtime)
        signed_train_parts.append(clip(signed_decode(base_oof, m.predict(zm), .9)))
        signed_parts.append(clip(signed_decode(base_test, m.predict(zmt), .9)))
    signed_raw_train, signed_raw = np.mean(signed_train_parts, axis=0), np.mean(signed_parts, axis=0)

    # Full-raw signed XGB diversity.
    full_cols = list(mo.columns)
    zfull, zfullt = numeric_frame(mo, full_cols), numeric_frame(mt, full_cols)
    full_train_parts, full_parts = [], []
    for seed in (cfg.seed + 3201, cfg.seed + 3202):
        m = fit_xgb(xgb_model(cfg.signed_trees, seed, cfg, 5), zfull, signed_target, w, runtime)
        full_train_parts.append(clip(signed_decode(base_oof, m.predict(zfull), .9)))
        full_parts.append(clip(signed_decode(base_test, m.predict(zfullt), .9)))
    full_raw_train, full_raw = np.mean(full_train_parts, axis=0), np.mean(full_parts, axis=0)

    # Exactly top-K additional raw columns: meta/stage/ratio columns are excluded first.
    selector = fit_xgb(xgb_model(cfg.selector_trees, cfg.seed + 3301, cfg, 5), zfull, signed_target, w, runtime)
    importance = pd.Series(selector.feature_importances_, index=full_cols).sort_values(ascending=False)
    excluded = set(meta_cols) | {c for c in full_cols if c.startswith("stage_") or c.startswith("anchor_lr_")}
    genuine_raw_columns = set(x.columns)
    raw_candidates = [c for c in importance.index if c in genuine_raw_columns and c not in excluded]
    selected_raw = raw_candidates[:min(cfg.top_k, len(raw_candidates))]
    selected_cols = list(meta_cols) + selected_raw
    if len(selected_cols) - len(meta_cols) != len(selected_raw): raise AssertionError("selected raw columns are not additional")
    if not set(selected_raw).issubset(genuine_raw_columns): raise AssertionError("derived meta columns entered raw selection")
    selected_cats = [c for c in selected_cols if str(mo[c].dtype) == "category"]
    selected_train_parts, selected_parts = [], []
    for seed in (cfg.seed + 3401, cfg.seed + 3402, cfg.seed + 3403):
        p = lgb_params(cfg.selected_trees, seed, cfg, "mae", 127, 180); p["learning_rate"] = .025
        m = fit_lgb("regressor", p, mo[selected_cols], signed_target, w, selected_cats, runtime)
        selected_train_parts.append(clip(signed_decode(base_oof, m.predict(mo[selected_cols]), .85)))
        selected_parts.append(clip(signed_decode(base_test, m.predict(mt[selected_cols]), .85)))
    selected_raw_train, selected_raw_pred = np.mean(selected_train_parts, axis=0), np.mean(selected_parts, axis=0)

    a = cfg.endpoint_alpha
    direct_endpoint_train, direct_endpoint = (1-a)*v3_train+a*direct_raw_train, (1-a)*v3+a*direct_raw
    signed_endpoint_train, signed_endpoint = (1-a)*v3_train+a*signed_raw_train, (1-a)*v3+a*signed_raw
    full_endpoint_train, full_endpoint = (1-a)*v3_train+a*full_raw_train, (1-a)*v3+a*full_raw
    selected_endpoint_train, selected_endpoint = (1-a)*v3_train+a*selected_raw_train, (1-a)*v3+a*selected_raw_pred
    signed_x150_train = clip(direct_endpoint_train + 1.5*(signed_endpoint_train-direct_endpoint_train))
    signed_x150 = clip(direct_endpoint + 1.5*(signed_endpoint-direct_endpoint))
    historical_base_train = clip(.8*signed_x150_train + .2*full_endpoint_train)
    historical_base = clip(.8*signed_x150 + .2*full_endpoint)
    final_train = clip(.9*historical_base_train + .1*selected_endpoint_train)
    final = clip(.9*historical_base + .1*selected_endpoint)

    components_oof = pd.DataFrame({ID_COL: train[ID_COL], "raw_order": raw_order, TARGET_COL: y, WEIGHT_COL: w, **{s: oof[s] for s in STAGES}}).sort_values("raw_order")
    components_test = pd.DataFrame({ID_COL: test[ID_COL], **{s: test_stages[s] for s in STAGES}})
    meta_oof = pd.DataFrame({ID_COL: train[ID_COL], "raw_order": raw_order, TARGET_COL: y, WEIGHT_COL: w,
        "anchor_v2": base_oof, "signed_target": signed_target, "training_residual_v3": v3_train,
        "training_direct_raw": direct_raw_train, "training_direct_endpoint": direct_endpoint_train,
        "training_signed_raw": signed_raw_train, "training_signed_endpoint": signed_endpoint_train,
        "training_signed_x150": signed_x150_train, "training_full_raw": full_raw_train,
        "training_full_endpoint": full_endpoint_train, "training_selected_raw": selected_raw_train,
        "training_selected_endpoint": selected_endpoint_train, "training_historical_base": historical_base_train,
        "training_final": final_train}).sort_values("raw_order")
    meta_test = pd.DataFrame({ID_COL: test[ID_COL], "anchor_v2": base_test, "residual_v3": v3,
        "direct_raw": direct_raw, "direct_endpoint": direct_endpoint, "signed_raw": signed_raw,
        "signed_endpoint": signed_endpoint, "signed_x150": signed_x150, "full_raw": full_raw,
        "full_endpoint": full_endpoint, "selected_raw": selected_raw_pred,
        "selected_endpoint": selected_endpoint, "historical_base": historical_base, "predict": final})
    report = {"stage_schema": STAGES, "folds": len(folds), "center": CENTER, "tail_thresholds": list(TAIL_THRESHOLDS),
              "blend": {"endpoint_alpha": a, "signed_extrapolation": 1.5,
                        "historical_base": {"signed_x150": .8, "full_endpoint": .2},
                        "final": {"historical_base": .9, "selected_endpoint": .1}},
              "selected_features": selected_cols, "selected_additional_raw_features": selected_raw,
              "selected_additional_raw_count": len(selected_raw), "top_k": cfg.top_k,
              "training_diagnostics": {"anchor_v2_crossfit_wmae": weighted_mae(y, base_oof, w),
                                       "training_component_warning": "second-level component columns are fitted on the OOF-stage frame and are training diagnostics, not strict outer predictions"},
              "feature_policy": "all transforms and selectors are fitted from the current training input only; inference rows only receive transform()",
              "date_policy": "dt is excluded from model features and used only for relative age and optional latest-date evaluation"}
    return final, report, {"oof_structural_stages.csv": components_oof, "test_structural_stages.csv": components_test,
                            "oof_meta_components.csv": meta_oof, "test_meta_components.csv": meta_test}


def validate_inputs(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if not {ID_COL, TARGET_COL}.issubset(train): raise ValueError("train requires id and target")
    if ID_COL not in test: raise ValueError("test requires id")
    if len(train) < 8 or len(test) == 0: raise ValueError("need at least 8 train rows and one test row")
    if train[ID_COL].isna().any() or test[ID_COL].isna().any(): raise ValueError("id must not contain null values")
    if train[ID_COL].duplicated().any() or test[ID_COL].duplicated().any(): raise ValueError("id must be unique")
    y = to_numeric(train[TARGET_COL]).to_numpy(float)
    if not np.isfinite(y).all() or np.any(y < 0): raise ValueError("target must be finite and non-negative")
    if WEIGHT_COL in train:
        w = to_numeric(train[WEIGHT_COL]).to_numpy(float)
        if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0: raise ValueError("w must be finite, non-negative and positive in total")
    if ASOF_COL in train: parse_iso_date(train[ASOF_COL], ASOF_COL)
    if ASOF_COL in test: parse_iso_date(test[ASOF_COL], ASOF_COL)


def forward_validation(train: pd.DataFrame, cfg: Config) -> dict:
    if ASOF_COL not in train: return {"status": "skipped", "reason": "dt absent"}
    dates = parse_iso_date(train[ASOF_COL], ASOF_COL); groups = sorted(dates.dropna().unique())
    if len(groups) < 2: return {"status": "skipped", "reason": "fewer than two dates"}
    latest = groups[-1]; fit = dates < latest; val = dates == latest
    if fit.sum() < 8 or val.sum() == 0: return {"status": "skipped", "reason": "insufficient rows"}
    runtime = Runtime(cfg.device)
    pred, _, _ = run_graph(train.loc[fit].copy(), train.loc[val].drop(columns=[TARGET_COL, WEIGHT_COL], errors="ignore").copy(), cfg, runtime)
    y = to_numeric(train.loc[val, TARGET_COL]).to_numpy(float); w = to_numeric(train.loc[val, WEIGHT_COL]).to_numpy(float) if WEIGHT_COL in train else np.ones(val.sum())
    if float(w.sum()) <= 0: return {"status": "skipped", "reason": "latest date has zero total weight", "latest_date": pd.Timestamp(latest).strftime("%Y-%m-%d")}
    return {"status": "evaluated", "latest_date": pd.Timestamp(latest).strftime("%Y-%m-%d"), "train_rows": int(fit.sum()), "valid_rows": int(val.sum()), "wmae": weighted_mae(y, pred, w)}


def validate_generated_values(prediction: np.ndarray, frames: dict[str, pd.DataFrame], y: np.ndarray) -> None:
    lo, hi = float(np.min(y)), float(np.max(y))
    checks = {
        "oof_structural_stages.csv": STAGES,
        "test_structural_stages.csv": STAGES,
        "oof_meta_components.csv": ["anchor_v2", *[c for c in frames["oof_meta_components.csv"].columns if c.startswith("training_")]],
        "test_meta_components.csv": [c for c in frames["test_meta_components.csv"].columns if c != ID_COL],
    }
    arrays = [("final prediction", np.asarray(prediction, float))]
    for name, cols in checks.items():
        arrays.extend((f"{name}:{c}", frames[name][c].to_numpy(float)) for c in cols)
    for name, values in arrays:
        if not np.isfinite(values).all(): raise ValueError(f"Non-finite generated values in {name}")
        if np.any(values < lo - 1e-6) or np.any(values > hi + 1e-6):
            raise ValueError(f"Generated values outside training target bounds in {name}")


def run_pipeline(train: pd.DataFrame, test: pd.DataFrame, cfg: Config, artifact_dir: Path, output: Path, report_path: Path, do_validation: bool) -> dict:
    validate_inputs(train, test)
    if cfg.folds < 2: raise ValueError("folds must be at least 2")
    # Full-schema preflight occurs before any OOF/model work and catches conditional dt requirements.
    preflight = FeatureBuilder(); preflight.fit_transform(train); preflight.transform(test)
    started = time.time(); runtime = Runtime(cfg.device)
    prediction, graph_report, frames = run_graph(train, test, cfg, runtime)
    validate_generated_values(prediction, frames, to_numeric(train[TARGET_COL]).to_numpy(float))
    # Strict validation is deliberately completed before publishing any files.
    validation_report = forward_validation(train, cfg) if do_validation else {"status": "disabled"}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for name, frame in frames.items():
        path = artifact_dir / name; atomic_csv(frame, path); generated.append(str(path))
    submit = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), "predict": prediction}); atomic_csv(submit, output)
    report = {"pipeline": "self_contained_staged_structural_meta", "created_at": time.strftime("%F %T"), "config": asdict(cfg),
              "dependencies": "current train and test tables only", "generated_artifacts": generated,
              "runtime": {"seconds": time.time() - started, "requested_device": cfg.device,
                          "lgb_cpu_fallbacks": runtime.lgb_cpu_fallbacks, "xgb_cpu_fallbacks": runtime.xgb_cpu_fallbacks,
                          "catboost_cpu_fallbacks": runtime.catboost_cpu_fallbacks},
              "prediction": {"rows": len(prediction), "mean": float(np.mean(prediction)), "std": float(np.std(prediction)), "finite": bool(np.isfinite(prediction).all())},
              **graph_report}
    report["forward_validation"] = validation_report
    manifest = {"inputs": {"train_rows": len(train), "test_rows": len(test)}, "outputs": [str(output), str(report_path), *generated],
                "artifact_usage": "write-only audit outputs; no stage reads generated or historical prediction files", "stage_schema": STAGES}
    atomic_json(manifest, artifact_dir / "manifest.json"); report["generated_artifacts"].append(str(artifact_dir / "manifest.json"))
    atomic_json(report, report_path)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="train.csv"); p.add_argument("--test", default="test.csv")
    p.add_argument("--output", default="submit.csv"); p.add_argument("--report", default="solution_report.json")
    p.add_argument("--artifact-dir", default="pipeline_artifacts"); p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--folds", type=int, default=3); p.add_argument("--seed", type=int, default=2026); p.add_argument("--fast", action="store_true")
    p.add_argument("--no-forward-validation", action="store_true", help="Skip expensive strict latest-date recursive graph")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv); train, test = read_table(args.train), read_table(args.test)
    cfg = Config.fast(args.folds, args.seed, args.device) if args.fast else Config(folds=args.folds, seed=args.seed, device=args.device)
    log(f"Start staged pipeline train={len(train)} test={len(test)} fast={args.fast}")
    report = run_pipeline(train, test, cfg, Path(args.artifact_dir), Path(args.output), Path(args.report), not args.no_forward_validation)
    log(f"Saved {args.output}; runtime={report['runtime']['seconds']:.1f}s")


if __name__ == "__main__": main()
