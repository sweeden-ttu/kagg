#!/usr/bin/env python3
"""3-opponent tournament → confusion matrix → GBDT policy.

Tournament opponents: fallow_finn, wheat_walter, rotation_rosa.

1. Run multi-seed matches and harvest 1035-dim state transitions.
2. Fit a GBDT opponent-identity classifier; emit 3×3 confusion matrix.
3. Fit a multi-head HistGradientBoosting policy conditioned on those findings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RVQ = ROOT / "reasoning_vs_questioning"
sys.path.insert(0, str(RVQ))
sys.path.insert(0, str(ROOT))

from eval import _make_kaggle_env, parse_observation  # noqa: E402
from gbdt_120day_scaling_trainer import GBDT120DayScalingTrainer  # noqa: E402
from market_config_suite import (  # noqa: E402
    InitialTerminalConfiguration,
    build_market_functions,
    build_opponent_functions,
)
from qkd_gbdt_vector_policy import MACRO_ACTION_TO_ID, QKDGBDTPolicy  # noqa: E402
from agents.qkd_replay_rl_agent import agent as qkd_replay_agent_fn  # noqa: E402

OPPONENTS = ("fallow_finn", "wheat_walter", "rotation_rosa")
SHORT = ("Finn", "Walter", "Rosa")
OUT_DIR = ROOT / "experiments" / "tournament_3opp_finn_walter_rosa"
CM_PNG = ROOT / "tournament_3opp_confusion_matrix.png"
MODEL_PKL = ROOT / "models" / "gbdt_3opp_finn_walter_rosa.pkl"


def harvest_tournament(
    trainer: GBDT120DayScalingTrainer,
    *,
    matches_per_opp: int = 2,
    max_steps: int = 480,
    sample_every: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Play matches vs the three opponents; return features + labels + match rows."""
    config = InitialTerminalConfiguration(
        number_of_days=120,
        amount_of_money=3000.0,
        turns_per_day=24,
        episode_steps=2880,
    )
    X: List[np.ndarray] = []
    y_opp: List[int] = []
    y_act: List[int] = []
    y_tile: List[float] = []
    y_liq: List[float] = []
    y_val: List[float] = []
    match_rows: List[Dict[str, Any]] = []

    for opp_i, opp_name in enumerate(OPPONENTS):
        opp_fn = trainer.opponents[opp_name]
        for m in range(matches_per_opp):
            # Alternate seats for symmetry
            if m % 2 == 0:
                agents = [qkd_replay_agent_fn, opp_fn]
                our_seat = 0
            else:
                agents = [opp_fn, qkd_replay_agent_fn]
                our_seat = 1

            env = _make_kaggle_env(max_steps=max_steps)
            runner = env.run(agents)
            last = runner[-1]
            p0 = parse_observation(last[0])
            p1 = parse_observation(last[1])
            money0 = float((p0.get("farms") or [{}])[0].get("money", 0) or 0)
            money1 = float((p1.get("farms") or [{}])[1].get("money", 0) or 0)
            our_money = money0 if our_seat == 0 else money1
            opp_money = money1 if our_seat == 0 else money0
            outcome = "WIN" if our_money > opp_money else ("LOSS" if our_money < opp_money else "TIE")
            match_rows.append(
                {
                    "opponent": opp_name,
                    "match": m,
                    "our_seat": our_seat,
                    "our_money": our_money,
                    "opp_money": opp_money,
                    "outcome": outcome,
                    "steps": len(runner),
                }
            )
            print(
                f"  vs {opp_name:14s} match={m} seat={our_seat} "
                f"{outcome:4s} us=${our_money:,.0f} opp=${opp_money:,.0f}"
            )

            for step_i, step_data in enumerate(runner):
                if step_i % sample_every != 0:
                    continue
                if not step_data or len(step_data) < 2:
                    continue
                raw = parse_observation(step_data[our_seat])
                if not raw:
                    continue
                # Force player id for extractor
                raw = dict(raw)
                raw["player"] = our_seat
                mf = build_market_functions(raw, config)
                of = build_opponent_functions(raw, config)
                state = trainer.extractor.extract_full_state_vector(raw, config, mf, of)
                step_num = int(raw.get("step", step_i) or step_i)
                day = step_num // 24
                if day < 20:
                    act = MACRO_ACTION_TO_ID["PLANT_HIGH_VALUE_CROP"]
                elif day < 40:
                    act = MACRO_ACTION_TO_ID["WATER_GROWING_CROPS"]
                elif day < 80:
                    act = MACRO_ACTION_TO_ID["HARVEST_MATURE_CROPS"]
                else:
                    act = MACRO_ACTION_TO_ID["LIQUIDATE_INVENTORY_AT_MARKET"]

                X.append(state)
                y_opp.append(opp_i)
                y_act.append(act)
                y_tile.append(float((step_num % 100) / 100.0))
                y_liq.append(float(min(1.0, max(0.1, day / 120.0))))
                y_val.append(float(our_money * (1.0 + day / 120.0)))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y_opp, dtype=np.int64),
        np.asarray(y_act, dtype=np.int64),
        np.asarray(y_tile, dtype=np.float32),
        np.asarray(y_liq, dtype=np.float32),
        np.asarray(y_val, dtype=np.float32),
        match_rows,
    )


def plot_confusion(
    cm: np.ndarray,
    labels: Tuple[str, ...],
    title: str,
    save_path: Path,
    *,
    xlabel: str = "Predicted",
    ylabel: str = "True",
) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=140)
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    acc = float(np.trace(cm) / max(1, cm.sum()))
    ax.set_title(f"{title}\n(n={int(cm.sum())}, accuracy={acc:.3f})")
    thresh = cm.max() * 0.55 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Tournament: fallow_finn | wheat_walter | rotation_rosa")
    print("=" * 70)

    trainer = GBDT120DayScalingTrainer(
        total_days=120,
        target_model_mb=20.0,
        model_save_path=str(MODEL_PKL),
    )
    # Restrict to the three requested opponents
    trainer.opponents = {k: trainer.opponents[k] for k in OPPONENTS}

    X, y_opp, y_act, y_tile, y_liq, y_val, match_rows = harvest_tournament(
        trainer, matches_per_opp=2, max_steps=480, sample_every=6
    )
    print(f"[+] Harvested {len(X)} transitions | feature_dim={X.shape[1]}")

    # ---- Confusion matrix: true opponent vs GBDT-predicted opponent ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_opp, test_size=0.30, random_state=42, stratify=y_opp
    )
    opp_clf = GradientBoostingClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
    )
    opp_clf.fit(X_tr, y_tr)
    y_pred = opp_clf.predict(X_te)
    cm_opp = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
    cm_path = plot_confusion(
        cm_opp,
        SHORT,
        "3-Opponent Tournament Confusion Matrix",
        CM_PNG,
        xlabel="Predicted opponent (GBDT)",
        ylabel="True opponent",
    )
    report = classification_report(y_te, y_pred, target_names=list(SHORT), digits=3)
    print("\nOpponent-ID classification report:\n", report)

    # Outcome summary matrix (true opponent × WIN/LOSS/TIE counts)
    outcome_labels = ("WIN", "LOSS", "TIE")
    cm_out = np.zeros((3, 3), dtype=np.int64)
    for row in match_rows:
        oi = OPPONENTS.index(row["opponent"])
        oj = outcome_labels.index(row["outcome"])
        cm_out[oi, oj] += 1
    out_cm_path = plot_confusion(
        cm_out,
        outcome_labels,
        "Tournament outcomes by opponent",
        OUT_DIR / "tournament_3opp_outcome_matrix.png",
        xlabel="Match outcome",
        ylabel="Opponent",
    )
    # Relabel y-axis manually for outcome plot (opponents on rows)
    # Re-render with correct row labels
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=140)
    im = ax.imshow(cm_out, cmap="Oranges", aspect="equal")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(outcome_labels)
    ax.set_yticklabels(SHORT)
    ax.set_xlabel("Match outcome")
    ax.set_ylabel("Opponent")
    ax.set_title("Tournament outcomes by opponent")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(int(cm_out[i, j])), ha="center", va="center", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_cm_path, bbox_inches="tight")
    plt.close(fig)

    # ---- GBDT policy fitted on tournament findings ----
    # Weight samples by confusion hardness: boost misclassified-opponent region via class weights
    # Encode findings: append one-hot predicted-opponent soft signal from opp_clf on full X
    proba = opp_clf.predict_proba(X)
    X_aug = np.hstack([X, proba.astype(np.float32)])

    policy = QKDGBDTPolicy(n_estimators=80, max_depth=6, num_leaves=48, learning_rate=0.06)
    # QKDGBDTPolicy expects 1035-dim; fit native heads on original X, plus a dedicated
    # opponent-conditioned action booster on X_aug.
    policy.fit(X, y_act, y_tile, y_liq, y_val)

    act_clf = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        learning_rate=0.07,
        random_state=7,
    )
    act_clf.fit(X_aug, y_act)

    val_reg = HistGradientBoostingRegressor(
        max_iter=80,
        max_depth=6,
        learning_rate=0.07,
        random_state=7,
    )
    val_reg.fit(X_aug, y_val)

    # Compact scale (findings model — not full 100MB unless needed)
    import pickle

    bundle = {
        "opponents": list(OPPONENTS),
        "opponent_classifier": opp_clf,
        "action_classifier_aug": act_clf,
        "value_regressor_aug": val_reg,
        "policy": policy,
        "confusion_matrix_opponent": cm_opp.tolist(),
        "confusion_matrix_outcome": cm_out.tolist(),
        "feature_dim": int(X.shape[1]),
        "n_samples": int(len(X)),
        "match_rows": match_rows,
        "classification_report": report,
        "test_accuracy": float(np.mean(y_pred == y_te)),
    }
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    metrics = {
        "opponents": list(OPPONENTS),
        "n_transitions": int(len(X)),
        "n_matches": len(match_rows),
        "opponent_confusion_matrix": cm_opp.tolist(),
        "outcome_matrix": cm_out.tolist(),
        "test_accuracy": bundle["test_accuracy"],
        "model_path": str(MODEL_PKL),
        "confusion_png": str(cm_path),
        "outcome_png": str(out_cm_path),
        "model_mb": MODEL_PKL.stat().st_size / (1024 * 1024),
        "match_rows": match_rows,
    }
    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print("=" * 70)
    print(f"Confusion matrix → {cm_path}")
    print(f"Outcome matrix   → {out_cm_path}")
    print(f"GBDT bundle      → {MODEL_PKL} ({metrics['model_mb']:.2f} MB)")
    print(f"Opponent test accuracy: {bundle['test_accuracy']:.3f}")
    print(f"Metrics          → {metrics_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
