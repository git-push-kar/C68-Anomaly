"""A non-neural reference on the same evidence the adapter reads.

The adapter is scored on a prompt. If a gradient-boosted model on the *numbers in
that prompt* reaches a similar accuracy, then the task is mostly a tabular
decision and the language model is paying 2 B parameters for a lookup; if it
falls well short, the prose and the menu are doing real work. Either answer is
worth having before any test component is opened.

Design decisions, all of which favour the classifier so the comparison is
conservative:

* it sees only the menu-variant items -- the same items as the menu-variant
  validation slice -- because it has no way to read a menu and therefore cannot
  be asked an open-set episode;
* it is trained on **train events only**, with faults 16-20 folded into
  ``UNCLASSIFIED_OPEN_SET`` and fault-free events kept as ``NORMAL_TRANSIENT``,
  which is exactly what the model's targets say;
* every tag in the standard TEP set is a column, so the classifier is not helped
  by "which five sensors were listed" beyond the numbers themselves.

Usage::

    python scripts/evidence_classifier.py
    python scripts/evidence_classifier.py --out outputs/analysis/phase3a/evidence_classifier
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from llm.benchmark import evaluate_v2 as ev  # noqa: E402
from llm.benchmark.label_scorer import candidates  # noqa: E402

TRAIN = ROOT / "outputs/benchmark_v2/dataset/train.jsonl"
VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
DEFAULT_OUT = ROOT / "outputs/analysis/phase3a/evidence_classifier"

#: The standard TEP variable set. Fixed rather than derived from the data, so a
#: column means the same thing in train and validation.
TAGS = [f"XMEAS_{n}" for n in range(1, 42)] + [f"XMV_{n}" for n in range(1, 12)]
GLOBALS = ["overall_score", "lstm_error", "lstm_ratio", "cva_q", "cva_ratio",
           "lag", "slow_drift", "n_listed", "fused_both", "fused_dynamic", "fused_level"]
#: Per tag: z, deviation, trend, onset minutes, residual share, present.
PER_TAG = 6


def load(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def gold_of(item: Dict) -> str:
    target = item.get("target") or {}
    return str(target.get("fault_id") or item.get("label") or "")


def _num(pattern: str, text: str, default: float = 0.0) -> float:
    import re

    match = re.search(pattern, text)
    return float(match.group(1)) if match else default


def prompt_features(prompt: str) -> List[float]:
    """The evidence block as a fixed-length vector.

    Parsed from the rendered prompt, not from the event object, so the classifier
    and the adapter are reading the same page.
    """
    import re

    values: Dict[Tuple[str, str], float] = {}
    listed = 0
    for line in prompt.splitlines():
        match = re.match(r"^- (.+?) \[(X(?:MEAS|MV)_\d{1,2}), [^\]]+\]: z=(-?[\d.]+), "
                         r"deviation (-?[\d.]+)%, trend (\w+)", line)
        if not match:
            continue
        tag = match.group(2)
        z = float(match.group(3))
        deviation = float(match.group(4))
        trend = {"increasing": 1.0, "decreasing": -1.0}.get(match.group(5).lower(), 0.0)
        values[(tag, "z")] = z
        values[(tag, "dev")] = deviation
        values[(tag, "trend")] = trend
        values[(tag, "present")] = 1.0
        listed += 1
    for line in prompt.splitlines():
        match = re.match(r"^\d+\. .+? \[(X(?:MEAS|MV)_\d{1,2}), [^\]]+\] at \+([\d.]+) min", line)
        if match:
            values[(match.group(1), "onset")] = float(match.group(2))
        match = re.match(r"^- (X(?:MEAS|MV)_\d{1,2}) \[[^\]]+\]: residual share ([\d.]+)", line)
        if match:
            values[(match.group(1), "residual")] = float(match.group(2))

    row = [
        _num(r"Overall anomaly score: (-?[\d.]+)", prompt),
        _num(r"reconstruction error (-?[\d.]+) ", prompt),
        _num(r"reconstruction error [\d.]+ \(threshold [\d.]+, ratio ([\d.]+)x", prompt),
        _num(r"CVA Q-statistic (-?[\d.]+) ", prompt),
        _num(r"CVA Q-statistic [\d.]+ \(threshold [\d.]+, ratio ([\d.]+)x", prompt),
        _num(r"Dominant lag (\d+) window", prompt),
        _num(r"slow-drift ratio ([\d.]+)", prompt),
        float(listed),
    ]
    fused = ""
    match = re.search(r"Fused trigger: (.+)", prompt)
    if match:
        fused = match.group(1).lower()
    row += [
        1.0 if "both" in fused else 0.0,
        1.0 if "dynamic" in fused else 0.0,
        1.0 if "level" in fused else 0.0,
    ]
    for tag in TAGS:
        row += [values.get((tag, key), 0.0) for key in
                ("z", "dev", "trend", "onset", "residual", "present")]
    return row


def dataset(rows: Sequence[Dict]) -> Tuple[np.ndarray, List[str], List[str], List[Dict]]:
    matrix = np.asarray([prompt_features(row["prompt"]) for row in rows], dtype=np.float32)
    golds = [gold_of(row) for row in rows]
    ids = [str(row.get("item_id")) for row in rows]
    return matrix, golds, ids, list(rows)


def cluster_of(item: Dict) -> str:
    return "|".join(str(item.get(key)) for key in ("source_file", "fault", "simulationRun"))


def rows_for(preds: Sequence[str], probs: Sequence[Dict[str, float]], golds: Sequence[str],
             ids: Sequence[str], items: Sequence[Dict]) -> List[Dict]:
    out = []
    for pred, prob, gold, item_id, item in zip(preds, probs, golds, ids, items):
        out.append({"item_id": item_id, "gold": gold, "pred": pred, "probs": dict(prob),
                    "run": cluster_of(item)})
    return out


def top_k(probs: Dict[str, float], k: int = 3) -> List[str]:
    return [label for label, _ in sorted(probs.items(), key=lambda pair: -pair[1])[:k]]


def fit_trees(x_train, y_train, x_val, seed: int = 42):
    import lightgbm as lgb

    labels = sorted(set(y_train))
    encoder = {label: index for index, label in enumerate(labels)}
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(labels), n_estimators=400,
        learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.9,
        subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, random_state=seed,
        verbose=-1,
    )
    model.fit(x_train, [encoder[label] for label in y_train])
    proba = model.predict_proba(x_val)
    ordered = [labels[index] for index in model.classes_]
    return [{label: float(p) for label, p in zip(ordered, row)} for row in proba]


def fit_mlp(x_train, y_train, x_val, seed: int = 42, epochs: int = 300):
    import torch
    import torch.nn as nn

    labels = sorted(set(y_train))
    encoder = {label: index for index, label in enumerate(labels)}
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True) + 1e-6
    xt = torch.tensor((x_train - mean) / scale, dtype=torch.float32)
    xv = torch.tensor((x_val - mean) / scale, dtype=torch.float32)
    yt = torch.tensor([encoder[label] for label in y_train], dtype=torch.long)
    model = nn.Sequential(nn.Linear(xt.shape[1], 128), nn.GELU(), nn.Dropout(0.2),
                          nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
                          nn.Linear(64, len(labels)))
    counts = Counter(y_train)
    weights = torch.tensor([len(y_train) / (len(labels) * counts[label]) for label in labels],
                           dtype=torch.float32)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(xt.shape[0], generator=generator)
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            optimiser.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimiser.step()
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(xv), dim=-1).numpy()
    return [{label: float(p) for label, p in zip(labels, row)} for row in proba]


def summarise(rows: Sequence[Dict], label: str) -> Dict:
    """The same metrics the adapter is reported on, so the rows are comparable."""
    report = ev.report(rows, label=label)
    return {
        "label": label,
        "items": report["items"],
        "accuracy": report["accuracy"],
        "top3": report["top3"],
        "macro_f1": report["macro_f1"],
        "abstention": report["abstention"],
        "per_class": {name: {"precision": info["precision"], "recall": info["recall"],
                             "f1": info["f1"], "support": info["support"]}
                      for name, info in report["per_fault"].items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter-preds", default=str(
        ROOT / "outputs/benchmark_v2/eval/preds_val_full_400.jsonl"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = load(TRAIN)
    val_rows = load(VAL)
    # Menu-variant only, on both sides: an open-set episode is a question about a
    # menu, and this model has no menu to read. Training on the removed-menu twins
    # would also give identical evidence two different gold labels.
    train_rows = [row for row in train_rows if row.get("variant") == "menu"]
    val_rows = [row for row in val_rows if row.get("variant") == "menu"]

    def normalise(label: str) -> str:
        if label.startswith("FAULT_"):
            number = int(label.split("_")[-1])
            if 16 <= number <= 20:
                return "UNCLASSIFIED_OPEN_SET"
        return label

    x_train, y_train, _ids, _items = dataset(train_rows)
    y_train = [normalise(label) for label in y_train]
    x_val, y_val, ids, items = dataset(val_rows)
    y_val = [normalise(label) for label in y_val]

    allowed = set(candidates())
    unknown = sorted(set(y_train) - allowed)
    if unknown:
        raise SystemExit(f"training labels outside the scorer's candidate set: {unknown}")

    print(json.dumps({"train_rows": len(y_train), "val_rows": len(y_val),
                      "features": int(x_train.shape[1]),
                      "train_labels": len(set(y_train)),
                      "val_labels": len(set(y_val))}, indent=2))

    trees = fit_trees(x_train, y_train, x_val, args.seed)
    trees_pred = [max(prob.items(), key=lambda pair: pair[1])[0] for prob in trees]
    trees_rows = rows_for(trees_pred, trees, y_val, ids, items)

    mlp = fit_mlp(x_train, y_train, x_val, args.seed)
    mlp_pred = [max(prob.items(), key=lambda pair: pair[1])[0] for prob in mlp]
    mlp_rows = rows_for(mlp_pred, mlp, y_val, ids, items)

    for name, rows in (("gbdt", trees_rows), ("mlp", mlp_rows)):
        (out / f"preds_{name}_val503.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    summary = {"gbdt": summarise(trees_rows, "gbdt"),
               "mlp": summarise(mlp_rows, "mlp")}

    # The adapter, restricted to the same item ids, so the three rows differ only
    # in the model.
    adapter_path = Path(args.adapter_preds)
    if adapter_path.exists():
        wanted = set(ids)
        adapter_rows = [row for row in load(adapter_path) if row["item_id"] in wanted]
        adapter_rows = [row for row in adapter_rows if not row.get("unobservable")]
        (out / "preds_adapter_val503.jsonl").write_text(
            "\n".join(json.dumps(row) for row in adapter_rows), encoding="utf-8")
        summary["adapter"] = summarise(adapter_rows, "adapter")
        summary["adapter_items"] = len(adapter_rows)
        # And on the intersection, because the adapter's file has no row for the
        # six unobservable items (faults 3 and 9) that the classifier can be scored
        # on: comparing 503 against 497 would quietly credit the classifier with
        # rows nobody else was asked about.
        common = wanted & {row["item_id"] for row in adapter_rows}
        summary["common_items"] = len(common)
        for name, rows in (("gbdt", trees_rows), ("mlp", mlp_rows)):
            summary[f"{name}_common"] = summarise(
                [row for row in rows if row["item_id"] in common], f"{name}:common")
        summary["adapter_common"] = summarise(
            [row for row in adapter_rows if row["item_id"] in common], "adapter:common")

    summary["class_counts_val"] = dict(Counter(y_val))
    summary["class_counts_train"] = dict(Counter(y_train))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                      encoding="utf-8")
    print(json.dumps({name: {key: summary[name][key] for key in
                             ("items", "accuracy", "top3", "macro_f1")}
                      for name in ("gbdt", "mlp", "adapter", "gbdt_common",
                                   "mlp_common", "adapter_common") if name in summary},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
