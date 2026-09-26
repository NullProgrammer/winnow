"""Fine-tune a token classifier on the synthetic corpus.

A plain PyTorch loop rather than HF `Trainer`, because the loss is the
interesting part and `Trainer` makes it awkward to reach:

**Two weights multiply, and both are necessary.**

1. `1/span_length`, precomputed in the dataset, so one ~500-token PEM block
   doesn't contribute as much gradient as ~100 emails. Without it the model
   learns "long base64 -> KEY", which is the shape prior we're trying to remove.
2. Inverse-frequency **class** weights, because `O` is ~95% of tokens and
   EMAIL/IP have ~3x the spans of the other four classes. Weighting only by
   span length would leave every entity drowned by `O`.

MPS specifics, all measured rather than assumed (see PLAN.md):
`attn_implementation="sdpa"` (FlashAttention-2 is CUDA-only),
`config.reference_compile = False` set via `AutoConfig` — it is *not* a
`from_pretrained` kwarg and raises TypeError if passed as one — fp32 because
MPS fp16 autocast has LayerNorm NaN issues, and seq len 512, which benchmarked
2.4x faster than 1024 for the same data.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenize_align import ENTITIES, IGNORE_INDEX, LABEL2ID, LABELS, align, decode_spans

MODEL_NAME = "answerdotai/ModernBERT-base"


@dataclass
class Config:
    train_path: Path = Path("data/synth/train.jsonl")
    val_path: Path = Path("data/synth/val.jsonl")
    out_dir: Path = Path("checkpoints/bio-modernbert")
    model_name: str = MODEL_NAME
    max_length: int = 512
    batch_size: int = 16
    epochs: int = 3
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    seed: int = 20260925
    limit: int | None = None
    val_limit: int | None = None
    markers: bool = False  # Phase 5 ablation; off for the flat BIO baseline
    log_every: int = 50


class WindowDataset(Dataset):
    """Windows stored as text + char spans, tokenized **once** at load time.

    The corpus on disk stays tokenizer-agnostic (Phase 4 compares ModernBERT
    against a code-pretrained control, which baked-in ids would prevent), but
    tokenizing lazily in `__getitem__` meant every window was re-tokenized and
    re-aligned on *every epoch* — three times the CPU work for identical
    output, serialised against the GPU. Precomputing once trades a few seconds
    of startup for that.

    `offsets` and `gold` are kept only when `keep_spans=True` (validation),
    since span-level scoring needs them and training does not.
    """

    def __init__(self, path: Path, tokenizer, max_length: int, limit: int | None = None,
                 markers: bool = False, keep_spans: bool = False, log=lambda *_: None):
        rows = []
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit is not None and i >= limit:
                    break
                rows.append(json.loads(line))

        self.keep_spans = keep_spans
        self.items: list[dict] = []
        t0 = time.perf_counter()

        for n, row in enumerate(rows, 1):
            prefix, shift = self._prefix(row, markers)
            text = prefix + row["text"]
            spans = [
                {"start": s["start"] + shift, "end": s["end"] + shift, "label": s["label"]}
                for s in row["spans"]
            ]
            a = align(text, spans, tokenizer, max_length)
            item = {
                "input_ids": a["input_ids"],
                "attention_mask": a["attention_mask"],
                "labels": a["labels"],
                "weights": a["weights"],
            }
            if keep_spans:
                item["offsets"] = a["offset_mapping"]
                item["gold"] = spans
                item["text"] = text
            self.items.append(item)
            if n % 2000 == 0:
                log(f"  tokenized {n}/{len(rows)} ({n / (time.perf_counter() - t0):.0f}/s)")

        self.spans_per_window = sum(len(r["spans"]) for r in rows) / max(1, len(rows))
        self.rows = rows

    @staticmethod
    def _prefix(row: dict, markers: bool) -> tuple[str, int]:
        """Phase 5's structural context, as plain text rather than new tokens.

        New special tokens would mean randomly-initialised embeddings trained on
        only ~30k examples; a plain-text prefix already tokenizes well.
        """
        if not markers:
            return "", 0
        test = "yes" if "_test.go" in row["path"] or "/testdata/" in row["path"] else "no"
        p = f"[go] [path:{row['path']}] [test:{test}]\n"
        return p, len(p)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate(batch: list[dict], pad_id: int) -> dict:
    # Pad to the batch's own max. Bucketing to multiples of 64 was tried on the
    # theory that MPS recompiles per tensor shape; it made no measurable
    # difference and only wasted attention on short batches, so it's gone.
    #
    # Profiled cost per step (batch 16, seq 512): forward 3.0 s, backward
    # 6.4 s, clip_grad_norm_ 2.7 s. The backward pass dominates, so there is no
    # data-pipeline fix available here.
    n = max(len(b["input_ids"]) for b in batch)

    def pad(key, fill):
        return torch.tensor([b[key][:n] + [fill] * (n - len(b[key][:n])) for b in batch])

    out = {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "labels": pad("labels", IGNORE_INDEX),
        "weights": torch.tensor(
            [b["weights"][:n] + [0.0] * (n - len(b["weights"][:n])) for b in batch],
            dtype=torch.float,
        ),
    }
    if "offsets" in batch[0]:
        out["meta"] = [
            {"offsets": b["offsets"], "gold": b["gold"], "text": b["text"]} for b in batch
        ]
    return out


def class_weights(ds: WindowDataset) -> torch.Tensor:
    """Inverse-sqrt-frequency weights over the label set.

    Inverse *sqrt* rather than plain inverse: plain inverse frequency on a label
    that is 95% of tokens produces an extreme ratio that destabilises early
    training. sqrt keeps the correction meaningful without that.
    """
    counts = Counter()
    for row in ds.rows:
        n_tokens = max(1, len(row["text"]) // 4)  # rough token estimate
        counts[LABEL2ID["O"]] += n_tokens
        for s in row["spans"]:
            if s["label"] in ENTITIES:
                counts[LABEL2ID[f"B-{s['label']}"]] += 1
                counts[LABEL2ID[f"I-{s['label']}"]] += max(0, len(s["text"]) // 4 - 1)

    total = sum(counts.values()) or 1
    w = torch.ones(len(LABELS))
    for i in range(len(LABELS)):
        c = counts.get(i, 0)
        w[i] = math.sqrt(total / c) if c > 0 else 1.0
    return w / w.mean()


def weighted_loss(logits, labels, weights, cls_w) -> torch.Tensor:
    """Per-token CE scaled by span weight AND class weight.

    Written with **no boolean mask indexing**. The obvious formulation —
    `flat_logits[valid]` — produces a tensor whose size depends on the data, so
    the host has to learn that size, forcing a device sync on every step. On
    MPS that measured ~10.5 s/step against a benchmarked 2.2 s. Masking
    multiplicatively keeps every shape static and the queue full.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    flat_w = weights.reshape(-1)

    valid = (flat_labels != IGNORE_INDEX).to(flat_logits.dtype)
    # clamp so ignored positions index a real class; `valid` zeroes them anyway.
    safe = flat_labels.clamp(min=0)

    ce = torch.nn.functional.cross_entropy(flat_logits, safe, reduction="none")
    w = flat_w * cls_w.to(logits.device)[safe] * valid
    return (ce * w).sum() / w.sum().clamp(min=1e-8)


def _match(pred: dict, gold: dict, exact: bool) -> bool:
    if pred["label"] != gold["label"]:
        return False
    if exact:
        return pred["start"] == gold["start"] and pred["end"] == gold["end"]
    return pred["end"] > gold["start"] and pred["start"] < gold["end"]


@torch.no_grad()
def evaluate(model, loader, device, cls_w) -> dict:
    """Span-level P/R/F1 per entity, exact and overlap.

    Token-level F1 would look far better and mean nothing — a model that tags
    95% of tokens `O` correctly scores well while finding no PII at all.
    """
    model.eval()
    stats = {
        mode: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}) for mode in ("exact", "overlap")
    }
    total_loss, batches = 0.0, 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=ids, attention_mask=mask)
        total_loss += weighted_loss(out.logits, labels, batch["weights"].to(device), cls_w).item()
        batches += 1

        preds = out.logits.argmax(-1).cpu().tolist()
        for row, meta in zip(preds, batch["meta"]):
            offsets = meta["offsets"]
            predicted = decode_spans(offsets, row[: len(offsets)], meta["text"])
            gold = [g for g in meta["gold"] if g["label"] in ENTITIES]

            for mode in ("exact", "overlap"):
                used = set()
                for p in predicted:
                    hit = next(
                        (
                            i
                            for i, g in enumerate(gold)
                            if i not in used and _match(p, g, mode == "exact")
                        ),
                        None,
                    )
                    if hit is None:
                        stats[mode][p["label"]]["fp"] += 1
                    else:
                        used.add(hit)
                        stats[mode][p["label"]]["tp"] += 1
                for i, g in enumerate(gold):
                    if i not in used:
                        stats[mode][g["label"]]["fn"] += 1

    result = {"val_loss": total_loss / max(1, batches)}
    for mode in ("exact", "overlap"):
        per_entity = {}
        for entity in ENTITIES:
            s = stats[mode][entity]
            p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
            r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            per_entity[entity] = {"p": p, "r": r, "f1": f1, **s}
        result[mode] = per_entity
    return result


def build_model(cfg: Config):
    from transformers import AutoConfig, AutoModelForTokenClassification

    config = AutoConfig.from_pretrained(cfg.model_name, num_labels=len(LABELS))
    config.id2label = {i: l for i, l in enumerate(LABELS)}
    config.label2id = dict(LABEL2ID)
    # A config attribute, NOT a from_pretrained kwarg — passing it there raises
    # TypeError. ModernBERT torch.compiles its MLP by default and that path
    # misbehaves on MPS.
    config.reference_compile = False
    return AutoModelForTokenClassification.from_pretrained(
        cfg.model_name, config=config, attn_implementation="sdpa"
    )


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run(cfg: Config, log=print) -> dict:
    from transformers import AutoTokenizer

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    log("tokenizing (once, not per epoch) ...")
    train_ds = WindowDataset(cfg.train_path, tok, cfg.max_length, cfg.limit, cfg.markers,
                             keep_spans=False, log=log)
    val_ds = WindowDataset(cfg.val_path, tok, cfg.max_length, cfg.val_limit, cfg.markers,
                           keep_spans=True, log=log)
    log(f"train {len(train_ds)} windows | val {len(val_ds)} windows")

    def loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate(b, tok.pad_token_id or 0),
        )

    train_dl, val_dl = loader(train_ds, True), loader(val_ds, False)

    device = pick_device()
    model = build_model(cfg).to(device)
    cls_w = class_weights(train_ds)
    log(f"device {device} | params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, len(train_dl) * cfg.epochs)
    warmup = int(steps * cfg.warmup_frac)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: s / max(1, warmup) if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup)),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, running, running_t = time.perf_counter(), 0.0, None
        for step, batch in enumerate(train_dl, 1):
            loss = weighted_loss(
                model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits,
                batch["labels"].to(device),
                batch["weights"].to(device),
                cls_w,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

            # Accumulate on-device. Calling .item() every step forces a GPU
            # sync, which stalls the pipeline; syncing only at log time keeps
            # the queue full.
            running_t = loss.detach() if step == 1 else running_t + loss.detach()
            if step % cfg.log_every == 0:
                running = running_t.item()
                rate = step * cfg.batch_size / (time.perf_counter() - t0)
                log(f"  epoch {epoch} step {step}/{len(train_dl)} "
                    f"loss {running/step:.4f} {rate:.1f} ex/s")

        metrics = evaluate(model, val_dl, device, cls_w)
        macro = sum(v["f1"] for v in metrics["exact"].values()) / len(ENTITIES)
        email = metrics["exact"]["EMAIL"]["f1"]
        log(f"epoch {epoch}: val_loss {metrics['val_loss']:.4f} "
            f"macro-F1(exact) {macro:.3f} EMAIL-F1 {email:.3f} "
            f"[{(time.perf_counter()-t0)/60:.1f} min]")
        for entity, m in metrics["exact"].items():
            log(f"    {entity:9} P {m['p']:.2f} R {m['r']:.2f} F1 {m['f1']:.2f} "
                f"(tp {m['tp']} fp {m['fp']} fn {m['fn']})")

        history.append({"epoch": epoch, "macro_f1_exact": macro, **metrics})
        if macro > best:
            best = macro
            model.save_pretrained(cfg.out_dir)
            tok.save_pretrained(cfg.out_dir)
            log(f"    saved checkpoint (best macro-F1 {best:.3f})")

    (cfg.out_dir / "history.json").write_text(
        json.dumps({"config": {k: str(v) for k, v in cfg.__dict__.items()}, "history": history},
                   indent=2),
        encoding="utf-8",
    )
    return {"best_macro_f1_exact": best, "history": history}
