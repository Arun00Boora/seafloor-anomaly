"""
routers/evaluation.py

GET /evaluation             -> precision/recall at several review depths, and "review
                               effort saved", for each scoring method (as the test set is).
GET /evaluation/prevalence  -> the same questions when debris is RARE: all normal tiles
                               are kept and debris tiles are randomly thinned, many times.
"""

import numpy as np
from fastapi import APIRouter

from . import tiles as tiles_router  # reuse the loaded ranked index

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

REVIEW_FRACTIONS = [0.05, 0.10, 0.20, 0.30]   # review the top 5/10/20/30% of tiles
RECALL_TARGETS = [0.50, 0.80, 0.90]           # share of all debris we want to find
PREVALENCE_RATES = [0.10, 0.05, 0.02]         # share of tiles that are debris
PREVALENCE_TRIALS = 300                       # random draws averaged per rate

# (key used by the frontend, display label, field in the ranked index)
METHODS = [
    ("ensemble", "Ensemble", "ensemble_score"),
    ("ae", "Autoencoder", "ae_score"),
    ("baseline", "Brightness baseline", "baseline_score"),
    ("pc", "PatchCore (pretrained)", "pc_score"),
]


def is_debris(value) -> bool:
    """The label may be stored as 1/0, True/False or text; accept all of them."""
    return str(value).strip().lower() in ("1", "1.0", "true", "anomalous")


TIE_TRIALS = 200   # random orderings averaged, so tied scores get no unfair advantage


def evaluate_method(records: list[dict], score_field: str) -> dict:
    scores = np.array([r[score_field] for r in records], dtype=float)
    y = np.array([is_debris(r["true_label"]) for r in records])
    n = len(y)
    total_debris = int(y.sum())
    rng = np.random.default_rng(0)

    found_at = [[] for _ in REVIEW_FRACTIONS]    # debris found in top-k, per trial
    needed_k = [[] for _ in RECALL_TARGETS]      # tiles needed for each recall target, per trial

    for _ in range(TIE_TRIALS):
        # Sort by score (high to low). Tiles with EXACTLY equal scores are ordered randomly.
        # np.lexsort treats the LAST key as the primary one.
        order = np.lexsort((rng.random(n), -scores))
        cum = np.cumsum(y[order])
        for j, fraction in enumerate(REVIEW_FRACTIONS):
            k = max(1, round(n * fraction))
            found_at[j].append(cum[k - 1])
        for j, target in enumerate(RECALL_TARGETS):
            needed_k[j].append(int(np.argmax(cum >= target * total_debris)) + 1)

    at_depth = []
    for j, fraction in enumerate(REVIEW_FRACTIONS):
        k = max(1, round(n * fraction))
        found = float(np.mean(found_at[j]))
        at_depth.append({
            "fraction": fraction,
            "tiles_reviewed": k,
            "debris_found": round(found),
            "precision": found / k,
            "recall": found / total_debris if total_debris else 0.0,
        })

    effort = []
    for j, target in enumerate(RECALL_TARGETS):
        k = float(np.mean(needed_k[j]))
        effort.append({
            "recall_target": target,
            "tiles_reviewed": round(k),
            "fraction_reviewed": k / n,
            "effort_saved": 1 - k / n,
        })

    return {"at_depth": at_depth, "effort": effort}

@router.get("")
def get_evaluation():
    records = tiles_router._RANKED_TILES
    n = len(records)
    n_debris = sum(is_debris(r["true_label"]) for r in records)

    return {
        "n_tiles": n,
        "n_debris": n_debris,
        "debris_rate": n_debris / n if n else 0.0,
        "methods": [
            {"key": key, "label": label, **evaluate_method(records, field)}
            for key, label, field in METHODS
        ],
    }


_prevalence_cache = None


def compute_prevalence() -> dict:
    """Keep every normal tile, randomly keep only enough debris tiles to hit each target
    rate, rank, measure, and repeat. A fixed seed makes the numbers reproducible."""
    records = tiles_router._RANKED_TILES
    rng = np.random.default_rng(0)

    labels = np.array([is_debris(r["true_label"]) for r in records])
    scores = {key: np.array([r[field] for r in records], dtype=float) for key, _, field in METHODS}
    debris_idx = np.flatnonzero(labels)
    normal_idx = np.flatnonzero(~labels)

    rates_out = []
    for rate in PREVALENCE_RATES:
        n_debris = max(1, round(rate / (1 - rate) * len(normal_idx)))
        n_total = len(normal_idx) + n_debris

        # collected[key] = {"prec": [per-depth lists], "rec": [...], "eff": [per-target lists]}
        collected = {
            key: {
                "prec": [[] for _ in REVIEW_FRACTIONS],
                "rec": [[] for _ in REVIEW_FRACTIONS],
                "eff": [[] for _ in RECALL_TARGETS],
            }
            for key, _, _ in METHODS
        }

        for _ in range(PREVALENCE_TRIALS):
            chosen = np.concatenate([normal_idx, rng.choice(debris_idx, n_debris, replace=False)])
            y = labels[chosen]
            for key, _, _ in METHODS:
                # Tiny random jitter breaks exact score ties fairly (no method is favored by order).
                s = scores[key][chosen] + rng.random(len(chosen)) * 1e-9
                cum = np.cumsum(y[np.argsort(-s)])
                for j, f in enumerate(REVIEW_FRACTIONS):
                    k = max(1, round(n_total * f))
                    found = cum[k - 1]
                    collected[key]["prec"][j].append(found / k)
                    collected[key]["rec"][j].append(found / n_debris)
                for j, t in enumerate(RECALL_TARGETS):
                    k = int(np.argmax(cum >= t * n_debris)) + 1
                    collected[key]["eff"][j].append(1 - k / n_total)

        methods_out = []
        for key, label, _ in METHODS:
            c = collected[key]
            methods_out.append({
                "key": key,
                "label": label,
                "at_depth": [
                    {
                        "fraction": f,
                        "precision": float(np.mean(c["prec"][j])),
                        "precision_low": float(np.percentile(c["prec"][j], 5)),
                        "precision_high": float(np.percentile(c["prec"][j], 95)),
                        "recall": float(np.mean(c["rec"][j])),
                    }
                    for j, f in enumerate(REVIEW_FRACTIONS)
                ],
                "effort": [
                    {"recall_target": t, "effort_saved": float(np.mean(c["eff"][j]))}
                    for j, t in enumerate(RECALL_TARGETS)
                ],
            })

        rates_out.append({
            "rate": rate,
            "n_debris": n_debris,
            "n_total": n_total,
            "methods": methods_out,
        })

    return {
        "n_normal": int(len(normal_idx)),
        "trials": PREVALENCE_TRIALS,
        "rates": rates_out,
    }


@router.get("/prevalence")
def get_prevalence():
    global _prevalence_cache
    if _prevalence_cache is None:
        _prevalence_cache = compute_prevalence()
    return _prevalence_cache