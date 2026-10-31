"""Four ways to read the same answer field, measured against each other.

The label scorer reads `{"fault_id": "` and then asks which candidate string the
model prefers. That question has more than one defensible answer, and the last two
checks showed the answer matters enough to move a selection:

``a. mean``      mean log-probability per token (what the report now quotes)
``b. joint``     the summed log-probability, i.e. the string's own probability
``c. decode``    greedy-decode the fault_id value and take what comes out, with the
                 chain-rule probability along the decoded path as its confidence
``d. hierarchical``  P(family) from the first token times P(number | FAULT)

(b) and (d) are the same quantity by the chain rule; the script measures that rather
than asserting it, because "the decomposition is not an independent instrument" is
itself the diagnosis.

Usage::

    venv_train/Scripts/python.exe scripts/instrument_check.py prefix --items-from disagreements.json
    venv_train/Scripts/python.exe scripts/instrument_check.py sweep
    venv_train/Scripts/python.exe scripts/instrument_check.py fullval --step 300
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm.benchmark import evaluate_v2 as ev  # noqa: E402
from llm.benchmark.label_scorer import (  # noqa: E402
    ANSWER_PREFIX,
    _causal_mask,
    _forward,
    _suffix_ids,
    candidates,
    prompt_for,
)

VAL = ROOT / "outputs/benchmark_v2/dataset/val.jsonl"
ADAPTER = ROOT / "outputs/tep_rca_v2_protocolA_seed42"
OUT = ROOT / "outputs/analysis/phase3a/instrument_check"
CONFIG = ROOT / "configs/config_v2_protocolA_seed42.yaml"
SUBSET = ROOT / "outputs/benchmark_v2/eval/val_subsample_ids.json"
FAMILY_OF = {label: ("FAULT" if label.startswith("FAULT_")
                     else "NORMAL" if label == "NORMAL_TRANSIENT"
                     else "UNCLASSIFIED")
             for label in candidates()}


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def gold_of(item: Dict) -> str:
    return str((item.get("target") or {}).get("fault_id") or "")


def stratify_subset(items: Sequence[Dict]) -> List[Dict]:
    wanted = json.loads(SUBSET.read_text(encoding="utf-8"))
    by_id = {item["item_id"]: item for item in items}
    return [by_id[item_id] for item_id in wanted if item_id in by_id]


# --------------------------------------------------------------------------- #
def score_detail(model, tokenizer, questions: Sequence[str], dummy: int,
                 batch: int = 16, via: str = "language_model") -> List[Dict]:
    """Every candidate's token path, with the per-token log-probabilities kept.

    One prefill per batch of items, then each label costs a few cached steps for all
    rows at once -- the same shape as ``label_scorer.score_batch``, but it returns
    the path instead of a single number so the four instruments can be derived from
    one measurement.
    """
    import torch

    labels = candidates()
    suffixes = {label: _suffix_ids(tokenizer, label) for label in labels}
    decode = {value: tokenizer.decode([value]) for value in
              {token for path in suffixes.values() for token in path}}
    results: List[Dict] = []
    for start in range(0, len(questions), batch):
        chunk = list(questions[start:start + batch])
        encoded = [tokenizer(prompt_for(question) + ANSWER_PREFIX,
                             add_special_tokens=True)["input_ids"] for question in chunk]
        width = max(len(ids) for ids in encoded)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        inputs, masks = [], []
        for ids in encoded:
            pad = width - len(ids)
            inputs.append(torch.tensor([pad_id] * pad + list(ids)))
            masks.append(torch.tensor([0] * pad + [1] * len(ids)))
        ids = torch.stack(inputs).to(model.device)
        valid = torch.stack(masks).to(model.device)
        position_ids = (valid.long().cumsum(-1) - 1).clamp(min=0)
        image = torch.zeros(len(chunk), 3, dummy, dummy,
                            dtype=next(model.parameters()).dtype, device=model.device)
        flags = torch.zeros((len(chunk),), dtype=torch.long, device=model.device)

        started = time.time()
        with torch.no_grad():
            prefill = _forward(model, ids, valid, position_ids, None, image, flags, via)
            base_past = prefill.past_key_values
            base_logits = prefill.logits[:, -1, :]
            per_label: Dict[str, List[List[float]]] = {label: [] for label in labels}
            for label in labels:
                suffix = suffixes[label]
                step_past = copy.deepcopy(base_past)
                step_logits, step_valid = base_logits, valid
                step_positions = position_ids
                paths = [[] for _ in chunk]
                for token in suffix:
                    log_probs = torch.log_softmax(step_logits.float(), dim=-1)
                    chosen = log_probs[:, token].tolist()
                    for row, value in enumerate(chosen):
                        paths[row].append(float(value))
                    step_valid = torch.cat(
                        [step_valid, torch.ones((len(chunk), 1), dtype=step_valid.dtype,
                                                device=step_valid.device)], dim=1)
                    step_positions = (step_positions[:, -1:] + 1).to(step_positions.dtype)
                    step = _forward(model, torch.full((len(chunk), 1), token,
                                                      device=ids.device),
                                    step_valid, step_positions, step_past, image, flags, via)
                    step_past = step.past_key_values
                    step_logits = step.logits[:, -1, :]
                per_label[label] = paths
        seconds = time.time() - started
        for row in range(len(chunk)):
            detail = {}
            for label in labels:
                path = per_label[label][row]
                joint = float(sum(path))
                detail[label] = {"tokens": [decode[token] for token in suffixes[label]],
                                 "logps": [round(value, 6) for value in path],
                                 "joint": joint, "mean": joint / max(len(path), 1),
                                 "first": float(path[0])}
            results.append({"labels": detail, "seconds": round(seconds / len(chunk), 4)})
    return results


def decode_first_field(model, tokenizer, prompts: Sequence[str], dummy: int,
                       batch: int = 8, cap: int = 16,
                       via: str = "language_model") -> List[Dict]:
    """Greedy-decode the fault_id value, keeping the chain-rule probability.

    This is the deployed path in miniature: the answer field is read by generating it,
    not by scoring candidates. Batched over items with left padding, which is why the
    per-item cost is closer to the scoring path than to a 300-token generation.
    """
    import torch

    # The field ends with a quote, but the tokeniser does not make `"` its own
    # token: the model writes `FAULT_01",` in one step. Stopping on token ids
    # missed every row and decoded a full 16 tokens past the answer, so the stop is
    # decided on the decoded *text* and the quote-bearing token is not counted as
    # part of the label or of its chain-rule probability.
    legal = set(candidates())
    terminators = ('"', "\n", "<|im_end|>")
    stop_ids = {tokenizer.convert_tokens_to_ids(token) for token in terminators}
    stop_ids = {value for value in stop_ids if isinstance(value, int) and value >= 0}
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    rows: List[Dict] = []
    for start in range(0, len(prompts), batch):
        chunk = list(prompts[start:start + batch])
        encoded = [tokenizer(prompt_for(question) + ANSWER_PREFIX,
                             add_special_tokens=True)["input_ids"] for question in chunk]
        width = max(len(ids) for ids in encoded)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        inputs, masks = [], []
        for ids in encoded:
            pad = width - len(ids)
            inputs.append(torch.tensor([pad_id] * pad + list(ids)))
            masks.append(torch.tensor([0] * pad + [1] * len(ids)))
        ids = torch.stack(inputs).to(model.device)
        valid = torch.stack(masks).to(model.device)
        positions = (valid.long().cumsum(-1) - 1).clamp(min=0)
        image = torch.zeros(len(chunk), 3, dummy, dummy,
                            dtype=next(model.parameters()).dtype, device=model.device)
        flags = torch.zeros((len(chunk),), dtype=torch.long, device=model.device)
        produced: List[List[int]] = [[] for _ in chunk]
        scores = [0.0 for _ in chunk]
        done = [False for _ in chunk]
        started = time.time()
        with torch.no_grad():
            out = _forward(model, ids, valid, positions, None, image, flags, via)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            for _step in range(cap):
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                next_ids = torch.argmax(logits, dim=-1).tolist()
                for row, token in enumerate(next_ids):
                    if done[row]:
                        continue
                    piece = tokenizer.decode([int(token)])
                    if int(token) in stop_ids or '"' in piece or "\n" in piece:
                        done[row] = True
                        continue
                    scores[row] += float(log_probs[row, token])
                    produced[row].append(int(token))
                if all(done):
                    break
                valid = torch.cat([valid, torch.ones((len(chunk), 1), dtype=valid.dtype,
                                                     device=valid.device)], dim=1)
                positions = (positions[:, -1:] + 1).to(positions.dtype)
                step_in = torch.tensor([[token] for token in next_ids], device=ids.device)
                out = _forward(model, step_in, valid, positions, past, image, flags, via)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
        seconds = time.time() - started
        for row in range(len(chunk)):
            text = tokenizer.decode(produced[row], skip_special_tokens=True).strip()
            label = text if text in legal else None
            rows.append({"text": text, "label": label, "raw": text,
                         "chain_logp": round(scores[row], 6),
                         "tokens": len(produced[row]),
                         "seconds": round(seconds / len(chunk), 4)})
    return rows


# --------------------------------------------------------------------------- #
def instrument_rows(items: Sequence[Dict], detail: Sequence[Dict],
                    decoded: Optional[Sequence[Dict]] = None) -> Dict[str, List[Dict]]:
    """Turn one measurement into the four instruments' prediction files."""
    labels = candidates()
    out: Dict[str, List[Dict]] = {"mean": [], "joint": [], "hierarchical": [], "decode": []}
    for index, (item, row) in enumerate(zip(items, detail)):
        paths = row["labels"]
        base = {"item_id": item["item_id"], "gold": gold_of(item),
                "run": "|".join(str(item.get(key)) for key in
                                ("source_file", "fault", "simulationRun")),
                "unobservable": bool(item.get("unobservable")),
                "variant": item.get("variant")}
        # (a) mean per token
        mean_scores = {label: paths[label]["mean"] for label in labels}
        best_mean = max(mean_scores, key=mean_scores.get)
        out["mean"].append({**base, "pred": best_mean, "value": mean_scores[best_mean],
                            "probs": _softmax(mean_scores), "seconds": row["seconds"]})
        # (b) joint
        joint_scores = {label: paths[label]["joint"] for label in labels}
        best_joint = max(joint_scores, key=joint_scores.get)
        out["joint"].append({**base, "pred": best_joint, "value": joint_scores[best_joint],
                             "probs": _softmax(joint_scores)})
        # (d) hierarchical: choose the family by its first-token mass, then the
        # label by its within-family conditional. Multiplying the two is the joint
        # again, which is why the *decision* is taken in two stages rather than on
        # the product: 15 FAULT labels each carry a fifth of the fault mass, so a
        # one-label family (NORMAL, UNCLASSIFIED) wins any comparison of per-label
        # joint probability whenever the fault number is uncertain.
        families: Dict[str, List[str]] = {}
        for label in labels:
            families.setdefault(FAMILY_OF[label], []).append(label)
        family_mass = {family: math.log(sum(math.exp(joint_scores[label]) for label in members))
                       for family, members in families.items()}
        winner_family = max(family_mass, key=family_mass.get)
        within = {label: joint_scores[label] - family_mass[winner_family]
                  for label in families[winner_family]}
        best_hier = max(within, key=within.get)
        out["hierarchical"].append({
            **base, "pred": best_hier,
            "value": family_mass[winner_family] + within[best_hier],
            "family": winner_family,
            "family_log_mass": family_mass[winner_family],
            "probs": _softmax({label: joint_scores[label] for label in labels})})
        # (c) decode
        if decoded is not None:
            row_c = decoded[index]
            out["decode"].append({**base, "pred": row_c["label"] or row_c["raw"],
                                  "value": row_c["chain_logp"], "raw": row_c["raw"],
                                  "decoded_tokens": row_c["tokens"],
                                  "seconds": row_c["seconds"]})
    return out


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    top = max(scores.values())
    weights = {label: math.exp(value - top) for label, value in scores.items()}
    norm = sum(weights.values()) or 1.0
    return {label: round(value / norm, 8) for label, value in weights.items()}


def metrics(rows: Sequence[Dict], label: str) -> Dict:
    report = ev.report(rows, label=label)
    return {key: report[key] for key in
            ("items", "accuracy", "top3", "macro_f1", "abstention", "per_fault")}


# --------------------------------------------------------------------------- #
def checkpoint_list() -> List[tuple]:
    from eval_protocolA import adapter_root, adapter_steps

    return adapter_steps(ADAPTER)


def entry_for(name: str) -> Dict:
    """The checkpoint-table row, read back from the four prediction files.

    Reading them instead of keeping the two passes' results in memory means a
    re-measurement of one instrument does not require re-running the others -- which
    is exactly what happened when the decode pass had to be fixed.
    """
    entry: Dict = {"step": int(name.replace("step", ""))}
    rows = {}
    for instrument in ("mean", "joint", "hierarchical", "decode"):
        path = OUT / f"preds_{instrument}_{name}.jsonl"
        if not path.exists():
            entry[instrument] = None
            continue
        predictions = load_jsonl(path)
        rows[instrument] = predictions
        report = metrics(predictions, f"{instrument}:{name}")
        entry[instrument] = {
            "accuracy": report["accuracy"], "top3": report["top3"],
            "macro_f1": report["macro_f1"],
            "abstention": report["abstention"],
            "fault15_recall": (report["per_fault"].get("FAULT_15") or {}).get("recall"),
            "normal_recall": (report["per_fault"].get("NORMAL_TRANSIENT") or {}).get("recall"),
            "seconds_per_item": round(
                sum(row.get("seconds") or 0 for row in predictions)
                / max(len(predictions), 1), 3),
            "invalid_decodes": (sum(1 for row in predictions if row.get("pred") not in
                                   set(candidates())) if instrument == "decode" else None),
        }
    if "joint" in rows and "hierarchical" in rows:
        entry["joint_vs_hierarchical_value_delta"] = round(max(
            abs(rows["joint"][i].get("value", 0.0)
                - rows["hierarchical"][i].get("value", 0.0))
            for i in range(min(len(rows["joint"]), len(rows["hierarchical"])))), 6)
    return entry


def cmd_sweep(args) -> int:
    from peft import PeftModel

    from eval_protocolA import bare, load_base

    OUT.mkdir(parents=True, exist_ok=True)
    observable = [item for item in load_jsonl(VAL) if not item.get("unobservable")]
    sample = stratify_subset(observable)
    questions = [item["prompt"] for item in sample]
    model, tokenizer, config, _applied = load_base(Path(args.config))
    dummy = int(config["llm"].get("dummy_image_size", 448))
    steps = checkpoint_list()
    first_step, first_path = steps[0]
    peft_model = PeftModel.from_pretrained(model, str(first_path),
                                           adapter_name=f"step{first_step}")
    summary = []
    for step, path in steps:
        name = f"step{step}"
        if name not in peft_model.peft_config:
            peft_model.load_adapter(str(path), adapter_name=name)
        peft_model.set_adapter(name)
        base_model = bare(peft_model)
        scored = all((OUT / f"preds_{instrument}_{name}.jsonl").exists()
                     for instrument in ("mean", "joint", "hierarchical"))
        if not (args.only_missing and scored):
            detail = score_detail(base_model, tokenizer, questions, dummy, args.batch)
            decoded = decode_first_field(base_model, tokenizer, questions, dummy,
                                         args.decode_batch, args.cap)
            rows = instrument_rows(sample, detail, decoded)
            for instrument, predictions in rows.items():
                (OUT / f"preds_{instrument}_{name}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in predictions), encoding="utf-8")
        elif not (OUT / f"preds_decode_{name}.jsonl").exists():
            decoded = decode_first_field(base_model, tokenizer, questions, dummy,
                                         args.decode_batch, args.cap)
            _write_decode(sample, decoded, name)
        entry = entry_for(name)
        summary.append(entry)
        print(json.dumps({key: (entry[key] or {}).get("macro_f1") if isinstance(entry[key], dict)
                          else entry[key] for key in entry}), flush=True)
    (OUT / "checkpoint_table_instruments.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    for instrument in ("mean", "joint", "hierarchical", "decode"):
        available = [row for row in summary if row.get(instrument)]
        best = max(available, key=lambda row: (row[instrument]["macro_f1"],
                                               row[instrument]["accuracy"], row["step"]))
        print(json.dumps({"instrument": instrument, "selected_step": best["step"],
                          "macro_f1": best[instrument]["macro_f1"],
                          "accuracy": best[instrument]["accuracy"]}), flush=True)
    return 0


def _write_decode(sample: Sequence[Dict], decoded: Sequence[Dict], name: str) -> None:
    legal = set(candidates())
    rows = []
    for item, row in zip(sample, decoded):
        rows.append({"item_id": item["item_id"], "gold": gold_of(item),
                     "run": "|".join(str(item.get(key)) for key in
                                     ("source_file", "fault", "simulationRun")),
                     "unobservable": bool(item.get("unobservable")),
                     "variant": item.get("variant"),
                     "pred": row["label"] or row["raw"], "raw": row["raw"],
                     "value": row["chain_logp"], "decoded_tokens": row["tokens"],
                     "seconds": row["seconds"],
                     "invalid": row["label"] is None and row["raw"] not in legal})
    (OUT / f"preds_decode_{name}.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def cmd_fullval(args) -> int:
    from peft import PeftModel

    from eval_protocolA import adapter_root, bare, load_base

    OUT.mkdir(parents=True, exist_ok=True)
    model, tokenizer, config, _applied = load_base(Path(args.config))
    dummy = int(config["llm"].get("dummy_image_size", 448))
    path = adapter_root(ADAPTER / f"checkpoint-{args.step}") \
        if args.step != "final" else adapter_root(ADAPTER)
    wrapped = PeftModel.from_pretrained(model, str(path), adapter_name="full")
    wrapped.set_adapter("full")
    base_model = bare(wrapped)
    items = load_jsonl(VAL)
    questions = [item["prompt"] for item in items]
    detail = score_detail(base_model, tokenizer, questions, dummy, args.batch)
    decoded = decode_first_field(base_model, tokenizer, questions, dummy,
                                 args.decode_batch, args.cap)
    rows = instrument_rows(items, detail, decoded)
    summary = {}
    for instrument in ("mean", "decode"):
        predictions = rows[instrument]
        tag = f"step{args.step}"
        (OUT / f"preds_{instrument}_{tag}_full.jsonl").write_text(
            "\n".join(json.dumps(row) for row in predictions), encoding="utf-8")
        observable = [row for row in predictions if not row["unobservable"]]
        unobservable = [row for row in predictions if row["unobservable"]]
        for name, subset in (("observable", observable), ("unobservable", unobservable)):
            summary[f"{instrument}:{name}"] = metrics(subset, f"{instrument}:{tag}:{name}")
        # Menu present vs removed: the split that says what the model does when the
        # true entry is not on the list. ``variant == "removed"`` is exactly the
        # open-set construction -- the fault's own menu line was deleted -- while
        # ``menu`` holds the faults whose entry is present plus the fault-free and
        # 16-20 rows, whose gold answer is UNCLASSIFIED_OPEN_SET either way.
        present_rows = [row for row in observable if row.get("variant") == "menu"]
        removed_rows = [row for row in observable if row.get("variant") != "menu"]
        summary[f"{instrument}:menu_present"] = metrics(present_rows,
                                                        f"{instrument}:{tag}:present")
        summary[f"{instrument}:menu_removed"] = metrics(removed_rows,
                                                        f"{instrument}:{tag}:removed")
    (OUT / f"fullval_{args.step}.json").write_text(json.dumps(summary, indent=2),
                                                   encoding="utf-8")
    print(json.dumps({key: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"],
                            "items": value["items"]}
                      for key, value in summary.items() if isinstance(value, dict)}, indent=1))
    return 0


def cmd_prefix(args) -> int:
    """The first-token and number-token probabilities behind each disagreement."""
    from peft import PeftModel

    from eval_protocolA import adapter_root, bare, load_base

    OUT.mkdir(parents=True, exist_ok=True)
    wanted = set(args.item_id or [])
    if args.items_from:
        payload = json.loads(Path(args.items_from).read_text(encoding="utf-8"))
        wanted |= {row["item_id"] for row in payload.get("items", [])}
    items = [item for item in load_jsonl(VAL) if item["item_id"] in wanted]
    if not items:
        raise SystemExit("no items matched")
    model, tokenizer, config, _applied = load_base(Path(args.config))
    dummy = int(config["llm"].get("dummy_image_size", 448))
    out = {"checkpoints": {}}
    for step in args.step:
        path = adapter_root(ADAPTER / f"checkpoint-{step}")
        wrapped = PeftModel.from_pretrained(model, str(path), adapter_name=f"p{step}")
        wrapped.set_adapter(f"p{step}")
        detail = score_detail(bare(wrapped), tokenizer, [item["prompt"] for item in items],
                              dummy, 1)
        rows = []
        for item, row in zip(items, detail):
            labels = row["labels"]
            families: Dict[str, float] = {}
            for label, path_row in labels.items():
                family = FAMILY_OF[label]
                mass = math.exp(path_row["joint"])
                families[family] = families.get(family, 0.0) + mass
            total = sum(families.values()) or 1.0
            numbers: Dict[str, Dict[str, float]] = {}
            for label, path_row in labels.items():
                if FAMILY_OF[label] != "FAULT":
                    continue
                numbers[label] = {
                    "number_tokens": path_row["tokens"][1:],
                    "p_number_given_fault": round(
                        math.exp(sum(path_row["logps"][1:])), 8),
                    "joint": round(math.exp(path_row["joint"]), 10),
                }
            ranked = sorted(labels.items(), key=lambda pair: -pair[1]["mean"])
            winner = ranked[0][0]
            family_argmax = max(families, key=families.get)
            rows.append({
                "item_id": item["item_id"], "gold": gold_of(item),
                "family_probability": {family: round(value / total, 6)
                                       for family, value in families.items()},
                "p_fault_then_numbers": numbers,
                "candidates": [{"label": label, "joint": round(math.exp(d["joint"]), 10),
                                "mean": round(d["mean"], 6)}
                               for label, d in ranked[:4]],
                "argmax_mean": winner,
                # The two numbers that decide whether the argmax is a decision or an
                # artefact of dividing by length: what the model thought of the
                # winner's *first* token, and which family it actually preferred.
                "argmax_first_token_probability": round(math.exp(labels[winner]["first"]), 8),
                "family_argmax": family_argmax,
                "argmax_family_matches": FAMILY_OF[winner] == family_argmax,
                "mean_gap_to_second": round(ranked[0][1]["mean"] - ranked[1][1]["mean"], 6),
            })
        out["checkpoints"][step] = rows
    (OUT / "prefix_diagnosis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1)[:3000])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--batch", type=int, default=16)
    sweep.add_argument("--decode-batch", dest="decode_batch", type=int, default=8)
    sweep.add_argument("--cap", type=int, default=16)
    sweep.add_argument("--only-missing", dest="only_missing", action="store_true",
                       help="reuse existing mean/joint/hierarchical predictions")
    sweep.set_defaults(func=cmd_sweep)

    full = sub.add_parser("fullval")
    full.add_argument("--step", required=True)
    full.add_argument("--batch", type=int, default=16)
    full.add_argument("--decode-batch", dest="decode_batch", type=int, default=8)
    full.add_argument("--cap", type=int, default=16)
    full.set_defaults(func=cmd_fullval)

    prefix = sub.add_parser("prefix")
    prefix.add_argument("--item-id", dest="item_id", action="append")
    prefix.add_argument("--items-from", dest="items_from")
    prefix.add_argument("--step", action="append", required=True)
    prefix.set_defaults(func=cmd_prefix)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
