"""Reproducible evaluation. Usage: python evaluate.py  -> reports/ (metrics.json, report.md, PNGs, CSVs)"""
from __future__ import annotations

import json
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nsre.config import FEATURES, MODEL_PATH, REPORTS_DIR, RISK_LEVELS, TARGET
from nsre.data import class_distribution, load_or_generate, make_invalid_records
from nsre.engine import ReasoningEngine
from nsre.hybrid import PolicyConfig, decide
from nsre.metrics import classification_summary, error_cases
from nsre.model import load_bundle, train_model, save_bundle
from nsre.validation import validate_record


def _cm_plot(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), RISK_LEVELS); ax.set_yticks(range(3), RISK_LEVELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _bar(labels, values, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_title(title); ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_or_generate()
    if MODEL_PATH.exists():
        bundle = load_bundle()
    else:
        bundle = train_model(df); save_bundle(bundle)
    engine = ReasoningEngine(bundle)
    test = df.loc[bundle["test_index"]].copy()
    train = df.loc[bundle["train_index"]]

    # 1. Neural performance
    neural_pred = engine.predictor.predict_batch(test)
    neural = classification_summary(test[TARGET], neural_pred)

    # Run full pipeline on every test row (valid by construction)
    assessments = [engine.assess(r) for r in test[FEATURES].to_dict("records")]
    test["neural"] = [a.neural.label for a in assessments]
    test["hybrid"] = [a.hybrid_risk for a in assessments]
    test["agreement_type"] = [a.decision.agreement_type for a in assessments]
    test["source"] = [a.decision.decision_source for a in assessments]
    test["needs_review"] = [a.decision.needs_review for a in assessments]
    test["rules"] = [",".join(a.decision.influencing_rules) for a in assessments]
    test["confidence"] = [round(a.neural.confidence, 3) for a in assessments]
    hybrid = classification_summary(test[TARGET], test["hybrid"])

    # 2. Rule frequency
    rule_counts = Counter(r.rule_id for a in assessments for r in a.fired_rules())
    rule_names = {r.rule_id: r.name for a in assessments[:1] for r in a.findings}
    n_fired_dist = Counter(len(a.fired_rules()) for a in assessments)
    multi = sum(v for k, v in n_fired_dist.items() if k >= 2)

    # 3. Agreement
    agree_dist = test["agreement_type"].value_counts(normalize=True).round(4).to_dict()
    n_agree = float(np.mean([a.decision.agree for a in assessments]))

    # 4. Hybrid distribution
    hybrid_dist = test["hybrid"].value_counts().reindex(RISK_LEVELS).fillna(0).astype(int).to_dict()
    source_dist = test["source"].value_counts().to_dict()
    changed = float((test["neural"] != test["hybrid"]).mean())

    # 5. Invalid-input detection (corrupted set vs the clean valid test rows)
    bad = make_invalid_records(df)
    detected = [not validate_record(r).valid for r in bad]
    false_alarm = [not validate_record(r).valid for r in test[FEATURES].to_dict("records")]
    leaks = [engine.assess(r).decision.hybrid_risk != "INVALID_INPUT" for r in bad]
    invalid = {"corrupted_records": len(bad), "detection_rate": float(np.mean(detected)),
               "valid_records_checked": len(false_alarm), "false_alarm_rate": float(np.mean(false_alarm)),
               "invalid_records_given_a_risk_class": int(sum(leaks))}

    # 6. Explanation coverage (valid cases only)
    def traceable(a):
        e, d = a.explanation, a.decision
        return bool(e) and d.decision_source in e and all(rid in e for rid in d.influencing_rules) \
            and a.neural.label in e
    coverage = float(np.mean([traceable(a) for a in assessments]))

    # 7. Error analysis
    ec = error_cases(test, TARGET, "hybrid")
    disagreements = test[test["agreement_type"] == "rules_higher"]
    cols = FEATURES + [TARGET, "neural", "hybrid", "confidence", "rules", "source"]
    ec["false_positives"][cols].to_csv(REPORTS_DIR / "cases_false_positives.csv", index=False)
    ec["false_negatives"][cols].to_csv(REPORTS_DIR / "cases_false_negatives.csv", index=False)
    disagreements[cols].head(15).to_csv(REPORTS_DIR / "cases_disagreements.csv", index=False)
    test[cols].to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    # Disagreement outcome: when rules overrode the model, was the escalation right?
    esc = test[test["source"] == "symbolic_escalation"]
    esc_stats = {"n": int(len(esc)),
                 "neural_correct": float((esc["neural"] == esc[TARGET]).mean()) if len(esc) else None,
                 "hybrid_correct": float((esc["hybrid"] == esc[TARGET]).mean()) if len(esc) else None}

    # 8. Runtime per prediction
    sample = test[FEATURES].to_dict("records")[:200]
    t0 = time.perf_counter()
    for r in sample:
        engine.assess(r)
    ms_full = (time.perf_counter() - t0) / len(sample) * 1000
    t0 = time.perf_counter()
    for r in sample:
        validate_record(r)
    ms_val = (time.perf_counter() - t0) / len(sample) * 1000


    # Policy sensitivity (ablation): vary thresholds, reuse stored neural output + rule findings
    sweep_rows = []
    for med, high, res in [(99, 99, 1.01), (3, 5, 1.01), (3, 5, 0.90), (3, 5, 0.75), (2, 4, 1.01), (2, 4, 0.90), (1, 3, 1.01)]:
        cfg = PolicyConfig(medium_floor_score=med, high_floor_score=high, confidence_resistance=res)
        preds = [decide(a.neural.label, a.neural.probabilities, a.findings, cfg) for a in assessments]
        hp = [d.hybrid_risk for d in preds]
        s_ = classification_summary(test[TARGET], hp)
        sweep_rows.append({"medium_floor": med, "high_floor": high, "confidence_resistance": res,
                           "escalations": sum(d.decision_source == "symbolic_escalation" for d in preds),
                           "review_flags": sum(d.needs_review for d in preds),
                           "accuracy": round(s_["accuracy"], 3), "macro_f1": round(s_["macro_f1"], 3),
                           "high_recall": round(s_["per_class"]["high"]["recall"], 3),
                           "high_precision": round(s_["per_class"]["high"]["precision"], 3)})
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(REPORTS_DIR / "policy_sweep.csv", index=False)

    # Plots
    _cm_plot(neural["confusion_matrix"], "Neural only", REPORTS_DIR / "cm_neural.png")
    _cm_plot(hybrid["confusion_matrix"], "Hybrid", REPORTS_DIR / "cm_hybrid.png")
    _bar([f"{k}\n{rule_names[k]}" for k in sorted(rule_counts)], [rule_counts[k] for k in sorted(rule_counts)],
         "Rule firing frequency (test set)", "cases", REPORTS_DIR / "rule_frequency.png")
    _bar(list(agree_dist), list(agree_dist.values()), "Model-rule agreement types", "share",
         REPORTS_DIR / "agreement.png")
    _bar(list(source_dist), list(source_dist.values()), "Hybrid decision source", "cases",
         REPORTS_DIR / "decision_sources.png")

    results = {
        "dataset": {"rows": len(df), "train": len(train), "test": len(test),
                    "class_distribution": class_distribution(df).to_dict()},
        "neural": {k: v for k, v in neural.items() if k != "report_text"},
        "hybrid_vs_truth": {k: v for k, v in hybrid.items() if k != "report_text"},
        "rule_counts": dict(rule_counts), "n_rules_fired_distribution": dict(n_fired_dist),
        "cases_with_multiple_rules": multi,
        "agreement_types": agree_dist, "no_conflict_rate": n_agree,
        "hybrid_distribution": hybrid_dist, "decision_sources": source_dist,
        "hybrid_changed_neural_class_rate": changed, "manual_review_flags": int(test["needs_review"].sum()),
        "escalation_outcomes": esc_stats, "invalid_input": invalid,
        "explanation_coverage": coverage,
        "policy_sweep": sweep_rows,
        "runtime_ms_per_prediction": {"full_pipeline": ms_full, "validation_only": ms_val},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=str))

    md = f"""# Evaluation Report (synthetic data, educational prototype)

> Results come from a synthetic dataset whose labels are a noisy function of the same features
> the model and rules see. They do NOT demonstrate real-world credit-prediction ability.

## Dataset
Rows: {len(df)} (train {len(train)}, test {len(test)}), stratified split, seed 42.

{class_distribution(df).to_markdown()}

## 1. Neural (Random Forest) performance on test set
Accuracy **{neural['accuracy']:.3f}**, macro-F1 **{neural['macro_f1']:.3f}**

```
{neural['report_text']}
```
Confusion matrix (rows=true, cols=pred; low/medium/high): `{neural['confusion_matrix']}`

## 2. Symbolic rule frequency (test set, n={len(test)})
{chr(10).join(f"- {k} ({rule_names[k]}): {v} ({v/len(test):.1%})" for k, v in sorted(rule_counts.items()))}

Cases where 2+ rules fired together: {multi} ({multi/len(test):.1%}). Distribution of #rules fired: {dict(sorted(n_fired_dist.items()))}

## 3. Model-rule agreement
{chr(10).join(f"- {k}: {v:.1%}" for k, v in agree_dist.items())}
- No-conflict rate (rules do not demand a higher class than the model): {n_agree:.1%}

## 4. Hybrid decision distribution
Classes: {hybrid_dist}. Sources: {source_dist}.
Hybrid changed the neural class in {changed:.1%} of cases; {int(test['needs_review'].sum())} flagged for manual review.

Hybrid vs ground truth: accuracy **{hybrid['accuracy']:.3f}**, macro-F1 **{hybrid['macro_f1']:.3f}**
(neural-only: {neural['accuracy']:.3f} / {neural['macro_f1']:.3f}).
High-class recall: neural {neural['per_class']['high']['recall']:.3f} -> hybrid {hybrid['per_class']['high']['recall']:.3f};
high-class precision: neural {neural['per_class']['high']['precision']:.3f} -> hybrid {hybrid['per_class']['high']['precision']:.3f}.

When rules escalated the model (n={esc_stats['n']}): neural was correct in {esc_stats['neural_correct']}, hybrid in {esc_stats['hybrid_correct']}.

## 5. Invalid-input detection
{json.dumps(invalid, indent=2)}

## 6. Explanation coverage (valid cases with a traceable explanation)
**{coverage:.1%}**. "Traceable" = explanation names the neural class, the decision source and every influencing rule ID.

## Policy sensitivity (first row = neural only, rules disabled; resistance 1.01 = never resist)
{sweep_df.to_markdown(index=False)}

Stricter floors escalate more cases; whether that helps depends entirely on how the (synthetic) labels were built.

## 7. Representative cases
See `cases_false_positives.csv`, `cases_false_negatives.csv`, `cases_disagreements.csv`.

## 8. Runtime
Full pipeline: **{ms_full:.2f} ms/prediction**; validation only: {ms_val:.4f} ms.

## Limitations
- Synthetic data; labels derived from the same features (circular by design). Rule thresholds are illustrative.
- Hybrid-vs-truth gains/losses reflect how well the made-up rules align with the made-up label formula.
- Single split, single seed; no confidence intervals. No fairness analysis; no real lending use.
"""
    (REPORTS_DIR / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
