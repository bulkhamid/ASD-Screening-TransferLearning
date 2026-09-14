"""Audit archived OOF predictions and export aggregate benchmark figures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.subject_metrics import compute_subject_metrics
from src.utils.subject_data import shuffled_training_labels


NAMES = {"simple2dcnn": "Simple2DCNN", "simple3dlstm_v2": "Simple3DLSTM v2", "r3d": "R3D"}
ABLATIONS = ("normal", "reversed", "shuffled", "static")


def load_subject_predictions(path, expected_subjects):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {"subject_id": row["subject_id"], "label": int(row["label"]), "logit": float(row["logit"])}
            for row in csv.DictReader(handle) if row["level"] == "subject"
        ]
    ids = [row["subject_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_subjects:
        raise ValueError(f"Prediction subject coverage does not match held-out fold: {path}")
    return rows


def summarize(root, baseline_path, output):
    config = json.loads((root / "configuration.json").read_text(encoding="utf-8"))
    saved = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    spec = {key: value for key, value in config.items() if key != "configuration_sha256"}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != config["configuration_sha256"] or digest != saved["configuration_sha256"]:
        raise ValueError("Configuration hash mismatch")
    with (root / "fold_assignments.csv").open(newline="", encoding="utf-8") as handle:
        assignments = list(csv.DictReader(handle))
    fold_tests = {}
    expected_labels = {}
    for fold in config["folds"]:
        roles = {role: {row["subject_id"] for row in assignments if int(row["fold"]) == fold and row["role"] == role}
                 for role in ("train", "validation", "test")}
        if roles["train"] & roles["test"] or roles["validation"] & roles["test"] or roles["train"] & roles["validation"]:
            raise ValueError("Archived fold has participant leakage")
        fold_tests[fold] = roles["test"]
    for row in assignments:
        label = int(row["label"])
        if row["subject_id"] in expected_labels and expected_labels[row["subject_id"]] != label:
            raise ValueError("Inconsistent archived participant label")
        expected_labels[row["subject_id"]] = label
    if sum(map(len, fold_tests.values())) != len(set().union(*fold_tests.values())):
        raise ValueError("Outer test folds overlap")

    report = {"configuration_sha256": digest, "completed_runs": len(saved["runs"]), "models": {}}
    for model in config["models"]:
        model_report = {"ablations": {}, "runs": []}
        for ablation in ABLATIONS:
            seed_metrics, logits_by_subject = {}, {}
            for seed in config["seeds"]:
                rows = []
                for fold in config["folds"]:
                    path = root / model / f"fold_{fold}" / f"seed_{seed}" / f"predictions_{ablation}.csv"
                    fold_rows = load_subject_predictions(path, fold_tests[fold])
                    if any(row["label"] != expected_labels[row["subject_id"]] for row in fold_rows):
                        raise ValueError("Held-out prediction labels differ from fold assignments")
                    rows.extend(fold_rows)
                seed_metrics[str(seed)] = compute_subject_metrics(rows)
                for row in rows:
                    logits_by_subject.setdefault(row["subject_id"], []).append(row["logit"])
            ensemble = compute_subject_metrics([
                {"subject_id": subject, "label": expected_labels[subject], "logit": float(np.mean(logits))}
                for subject, logits in logits_by_subject.items()
            ])
            aucs = [metrics["roc_auc"] for metrics in seed_metrics.values()]
            model_report["ablations"][ablation] = {
                "ensemble": ensemble, "per_seed": seed_metrics,
                "mean_seed_auc": float(np.mean(aucs)), "sd_seed_auc": float(np.std(aucs, ddof=1)),
            }
            if ablation == "normal" and not np.isclose(ensemble["roc_auc"], saved["models"][model]["subject_metrics"]["roc_auc"]):
                raise ValueError("Recomputed AUC differs from archived summary")
        model_report["auc_ci"] = saved["models"][model]["bootstrap_95_ci"]["roc_auc"]
        for fold in config["folds"]:
            for seed in config["seeds"]:
                directory = root / model / f"fold_{fold}" / f"seed_{seed}"
                history = json.loads((directory / "history.json").read_text())
                metrics = json.loads((directory / "metrics.json").read_text())
                budgets = config["protocol_config"]["models"][model]
                stages = budgets.get("stages", [{"name": "full", "epochs": budgets.get("max_epochs")}])
                stage_counts = {stage["name"]: sum(epoch["stage"] == stage["name"] for epoch in history) for stage in stages}
                model_report["runs"].append({
                    "fold": fold, "seed": seed, "epochs": len(history),
                    "stage_epochs": stage_counts,
                    "stages_stopped_early": [stage["name"] for stage in stages if stage_counts[stage["name"]] < stage["epochs"]],
                    "selection": metrics["selection"],
                })
        report["models"][model] = model_report
    report["baselines"] = json.loads(baseline_path.read_text())["results"]
    report["verification"] = "Recomputed every fold/seed/ablation from held-out subject predictions; checked labels, coverage, isolation, and configuration hash."
    output.mkdir(parents=True, exist_ok=True)
    (output / "reviewed_subject_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_figures(root, report, output)
    return report


def make_figures(root, report, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    models = list(NAMES)
    colors = ["#176d8f", "#a34142", "#397448"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, model, color in zip(axes, models, colors):
        values = report["models"][model]["ablations"]
        means = [values[name]["mean_seed_auc"] for name in ABLATIONS]
        errors = [values[name]["sd_seed_auc"] for name in ABLATIONS]
        ax.errorbar(range(4), means, yerr=errors, fmt="o-", color=color, capsize=3)
        ax.set(xticks=range(4), xticklabels=["Normal", "Reverse", "Shuffle", "Still"], ylim=(0.97, 1.003), title=NAMES[model])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean per-seed OOF AUC (zoomed axis)")
    fig.suptitle("Removing within-clip motion barely changes ranking", fontsize=13)
    fig.text(0.5, 0.01, "Error bars: SD across 3 seeds. Still = repeat each clip's middle frame; five clips remain per video.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output / "temporal_ablations.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, model in zip(axes, models):
        history = json.loads((root / model / "fold_1" / "seed_42" / "history.json").read_text())
        epochs = [row["global_epoch"] for row in history]
        ax.plot(epochs, [row["train_loss"] for row in history], label="Train clip BCE", color="#176d8f")
        ax.plot(epochs, [row["validation"]["nll"] for row in history], label="Validation subject NLL", color="#a34142")
        for previous, current in zip(history, history[1:]):
            if previous["stage"] != current["stage"]:
                ax.axvline(current["global_epoch"] - 0.5, color="#777777", linestyle=":")
        ax.set(yscale="log", xlabel="Epoch", title=NAMES[model])
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Learning curves: fixed example, fold 1 / seed 42", fontsize=13)
    fig.text(0.5, 0.01, "Train and validation losses use different aggregation levels. Dotted lines mark R3D stage changes.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, model in zip(axes, models):
        matrix = np.asarray(report["models"][model]["ablations"]["normal"]["ensemble"]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=53)
        for y in range(2):
            for x in range(2):
                ax.text(x, y, str(matrix[y, x]), ha="center", va="center", color="white" if matrix[y, x] > 25 else "black", fontsize=17)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Control", "ASD label"], yticklabels=["Control", "ASD label"], title=NAMES[model], xlabel="Predicted label")
    axes[0].set_ylabel("Dataset label")
    fig.suptitle("Held-out predictions: 102 videos, three-seed ensemble", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def summarize_permutations(primary_root, permutation_root, output):
    primary_config = json.loads((primary_root / "configuration.json").read_text())
    with (primary_root / "fold_assignments.csv").open(newline="", encoding="utf-8") as handle:
        assignments = list(csv.DictReader(handle))
    expected = {("simple2dcnn", seed) for seed in range(1042, 1047)} | {
        ("simple3dlstm_v2", 2042), ("r3d", 2042)
    }
    completed = {}
    for path in sorted(permutation_root.glob("*/summary.json")):
        config = json.loads((path.parent / "configuration.json").read_text())
        summary = json.loads(path.read_text())
        if config["cache_manifest_sha256"] != primary_config["cache_manifest_sha256"]:
            raise ValueError("Permutation used a different dataset manifest")
        if (path.parent / "fold_assignments.csv").read_bytes() != (primary_root / "fold_assignments.csv").read_bytes():
            raise ValueError("Permutation fold assignments differ from primary experiment")
        if config["seeds"] != [42] or config["protocol_config"] != primary_config["protocol_config"]:
            raise ValueError("Permutation controls must use seed 42 and the primary protocol")
        if len(summary["runs"]) != 5:
            raise ValueError("Permutation is missing one or more folds")
        for model, results in summary["models"].items():
            key = (model, config["permutation_seed"])
            if key not in expected or key in completed:
                raise ValueError("Unexpected or repeated permutation summary")
            pooled = []
            label_changes = []
            for fold in range(1, 6):
                train_rows = [row for row in assignments if int(row["fold"]) == fold and row["role"] == "train"]
                train_ids = {row["subject_id"] for row in train_rows}
                shuffled = shuffled_training_labels(train_rows, train_ids, key[1])
                if set(shuffled) != train_ids or sum(shuffled.values()) != sum(int(row["label"]) for row in train_rows):
                    raise ValueError("Reconstructed shuffle changed training coverage or class balance")
                label_changes.append({
                    "fold": fold, "training_subjects": len(train_rows),
                    "changed_labels": sum(shuffled[row["subject_id"]] != int(row["label"]) for row in train_rows),
                })
                control = path.parent / model / f"fold_{fold}" / "seed_42" / "predictions_normal.csv"
                original = primary_root / model / f"fold_{fold}" / "seed_42" / "predictions_normal.csv"
                def labels(csv_path):
                    with csv_path.open(newline="") as handle:
                        return {row["subject_id"]: int(row["label"]) for row in csv.DictReader(handle) if row["level"] == "subject"}
                expected_labels = labels(original)
                fold_rows = load_subject_predictions(control, set(expected_labels))
                if any(row["label"] != expected_labels[row["subject_id"]] for row in fold_rows):
                    raise ValueError("Permutation changed held-out labels or subject coverage")
                pooled.extend(fold_rows)
            recomputed = compute_subject_metrics(pooled)
            if not np.isclose(recomputed["roc_auc"], results["subject_metrics"]["roc_auc"]):
                raise ValueError("Recomputed permutation AUC differs from archived summary")
            completed[key] = {"model": model, "permutation_seed": key[1],
                              "reconstructed_training_label_changes": label_changes, **recomputed}
    if set(completed) != expected:
        raise ValueError(f"Permutation suite incomplete: {len(completed)}/{len(expected)}")
    output.mkdir(parents=True, exist_ok=True)
    rows = [completed[key] for key in sorted(completed)]
    report = {
        "training_seed": 42, "complete_fold_runs": 35, "controls": rows,
        "verification": "Same manifest, primary protocol, and folds; held-out labels unchanged; metrics recomputed from archived predictions.",
        "interpretation": "Training-label sanity controls with true validation labels retained for selection; not a formal permutation significance test.",
    }
    (output / "permutation_controls.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    primary = json.loads((output / "reviewed_subject_metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for index, model in enumerate(NAMES):
        values = [row["roc_auc"] for row in rows if row["model"] == model]
        ax.scatter(values, np.full(len(values), index), color="#7a858c", label="Shuffled training labels" if index == 0 else None, s=45)
        real = primary["models"][model]["ablations"]["normal"]["per_seed"]["42"]["roc_auc"]
        ax.scatter([real], [index], color="#176d8f", marker="D", label="Real labels, same seed" if index == 0 else None, s=55)
    ax.axvline(0.5, color="#999999", linestyle=":")
    ax.set(xlim=(0, 1.03), yticks=range(3), yticklabels=list(NAMES.values()), xlabel="Pooled out-of-fold ROC AUC", title="Training-label controls: same reviewed data and folds")
    ax.invert_yaxis()
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "permutation_controls.png", dpi=180)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--baselines", default="runs/subject_reviewed_shortcuts.json")
    parser.add_argument("--output_dir", default="docs/figures/reviewed_subject_benchmark")
    parser.add_argument("--permutations_dir", default=None)
    args = parser.parse_args()
    result = summarize(Path(args.run_dir), Path(args.baselines), Path(args.output_dir))
    if args.permutations_dir:
        summarize_permutations(Path(args.run_dir), Path(args.permutations_dir), Path(args.output_dir))
    print(json.dumps({"verified_runs": result["completed_runs"], "models": list(result["models"]), "output": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
