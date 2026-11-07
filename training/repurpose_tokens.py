#!/usr/bin/env python
"""Repurpose a model's reserved ``<unusedN>`` tokens as new single tokens.

The new-vocabulary recipe (a names-heavy domain whose names tokenize into
many shared subword pieces -- every tribe name ending in the same suffix,
every house in the same two syllables) is:

  1. rename spare ``<unusedN>`` tokens to the names, as *added tokens* so the
     tokenizer matches them whole in text (this script);
  2. train ONLY those embedding rows, initialized from a description
     (``--trainable-tokens`` / ``--trainable-tokens-init``), with the rest of
     the model frozen;
  3. then the ordinary SFT stage on top.

No embedding resize anywhere: the ids already exist, only their strings and
their rows change, so the quantized EXL3 model keeps working with the edited
tokenizer files (and ``merge_lora_bf16.py`` carries the rows into the bf16
model for a requantize).

What is edited (in a COPY of the tokenizer files unless ``--in-place``):
  * ``tokenizer.json``: the ``<unusedN>`` vocab entry is renamed to the new
    string and an ``added_tokens`` entry (``special: false``,
    ``normalized: false``) is added for it with the same id -- a plain BPE
    vocab entry alone is unreachable (no merge produces it), the added-token
    matcher is what makes the string a single token;
  * ``tokenizer_config.json``: the ``added_tokens_decoder`` entry;
  * ``added_tokens.json`` (if present): the string -> id map.

The edit is verified by re-tokenizing: each new string must encode to exactly
its id, alone and inside a sentence. Nothing is written if a check fails.

Usage:
    # see which <unused> tokens exist
    python training/repurpose_tokens.py --model /path/to/exl3-model --list

    # dry run (default): report the plan, verify, write nothing
    python training/repurpose_tokens.py --model /path/to/exl3-model \\
        --map '<unused6>=IceWing' '<unused7>=MudWing' '<unused8>=NightWing'

    # write to a copy of the model dir's tokenizer files (weights are NOT
    # copied; symlink or point the trainer at --tokenizer-dir if separate)
    python training/repurpose_tokens.py --model M --map-file names.json \\
        --out /path/to/tokenizer-out --write

    # or edit the model dir itself (a .bak of each file is kept)
    python training/repurpose_tokens.py --model M --map-file names.json --in-place --write

``--map-file`` is a JSON object ``{"<unused6>": "IceWing", ...}`` or a list of
``"<unused6>=IceWing"`` strings. Names are matched case-sensitively and
whole; a name that already tokenizes to one token is reported and skipped.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys


def load_map(args) -> dict[str, str]:
    pairs: dict[str, str] = {}
    entries = list(args.map or [])
    if args.map_file:
        with open(args.map_file, encoding="utf8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            pairs.update({str(k): str(v) for k, v in data.items()})
        else:
            entries += [str(e) for e in data]
    for e in entries:
        if "=" not in e:
            raise SystemExit(f"--map entry {e!r} must be OLD=NEW")
        old, new = e.split("=", 1)
        pairs[old.strip()] = new.strip()
    for old, new in pairs.items():
        if not new:
            raise SystemExit(f"empty replacement for {old!r}")
        if any(ch.isspace() for ch in new):
            raise SystemExit(f"{new!r}: a repurposed token must not contain whitespace")
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Model dir holding tokenizer.json")
    ap.add_argument("--map", nargs="*", default=None, metavar="OLD=NEW",
                    help="Renames, e.g. '<unused6>=IceWing'")
    ap.add_argument("--map-file", default=None, help="JSON {OLD: NEW} or [\"OLD=NEW\", ...]")
    ap.add_argument("--list", action="store_true", help="List the <unused*> tokens and exit")
    ap.add_argument("--out", default=None, help="Write the edited tokenizer files here")
    ap.add_argument("--in-place", action="store_true", help="Edit the model dir's files (keeps .bak)")
    ap.add_argument("--write", action="store_true", help="Actually write (default: dry run)")
    ap.add_argument("--allow-non-unused", action="store_true",
                    help="Allow renaming tokens that are not <unusedN> (dangerous)")
    args = ap.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise SystemExit("the 'tokenizers' package is required (pip install tokenizers)")

    tok_path = os.path.join(args.model, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise SystemExit(f"no tokenizer.json in {args.model}")
    with open(tok_path, encoding="utf8") as f:
        tj = json.load(f)
    model = tj["model"]
    vocab = model["vocab"]
    is_dict_vocab = isinstance(vocab, dict)
    if is_dict_vocab:
        piece_to_id = dict(vocab)
    else:   # Unigram: [[piece, score], ...]
        piece_to_id = {p: i for i, (p, _s) in enumerate(vocab)}
    added = tj.setdefault("added_tokens", [])
    added_by_content = {a["content"]: a for a in added}

    if args.list:
        unused = sorted(((i, p) for p, i in piece_to_id.items()
                         if re.fullmatch(r"<unused\d+>", p)), key=lambda x: x[0])
        for i, p in unused:
            print(f"{i:>8}  {p}")
        print(f"{len(unused)} <unused*> tokens")
        return

    pairs = load_map(args)
    if not pairs:
        raise SystemExit("nothing to do: give --map / --map-file (or --list)")
    if args.write and not (args.out or args.in_place):
        raise SystemExit("--write needs --out DIR or --in-place")

    base = Tokenizer.from_file(tok_path)
    plan = []
    for old, new in pairs.items():
        if old not in piece_to_id and old not in added_by_content:
            raise SystemExit(f"{old!r} is not a token")
        if not re.fullmatch(r"<unused\d+>", old) and not args.allow_non_unused:
            raise SystemExit(f"{old!r} is not an <unusedN> token (pass --allow-non-unused to force)")
        tid = added_by_content[old]["id"] if old in added_by_content else piece_to_id[old]
        if new in piece_to_id or new in added_by_content:
            raise SystemExit(f"{new!r} is already a token (id "
                             f"{piece_to_id.get(new, added_by_content.get(new, {}).get('id'))})")
        enc = base.encode(new, add_special_tokens=False).ids
        if len(enc) == 1:
            print(f"  skip {new!r}: already a single token ({enc[0]})")
            continue
        plan.append((old, new, tid, enc))
    if not plan:
        print("nothing to change")
        return
    print(f"{'old':<14}{'id':>8}  new             base tokenization")
    for old, new, tid, enc in plan:
        print(f"{old:<14}{tid:>8}  {new:<16}{enc}")

    # --- edit tokenizer.json ---
    for old, new, tid, _enc in plan:
        if old in piece_to_id:
            if is_dict_vocab:
                del vocab[old]
                vocab[new] = tid
            else:
                vocab[tid] = [new, vocab[tid][1]]
            piece_to_id.pop(old)
            piece_to_id[new] = tid
        if old in added_by_content:
            a = added_by_content.pop(old)
            a["content"] = new
            a["special"] = False
            a["normalized"] = False
            added_by_content[new] = a
        else:
            a = {"id": tid, "content": new, "single_word": False, "lstrip": False,
                 "rstrip": False, "normalized": False, "special": False}
            added.append(a)
            added_by_content[new] = a
    added.sort(key=lambda a: a["id"])

    # --- verify on the edited tokenizer (in memory) ---
    edited = Tokenizer.from_str(json.dumps(tj))
    ok = True
    for old, new, tid, _enc in plan:
        alone = edited.encode(new, add_special_tokens=False).ids
        sent = edited.encode(f"The {new} flew home.", add_special_tokens=False).ids
        good = alone == [tid] and sent.count(tid) == 1
        print(f"  verify {new!r}: alone -> {alone}, in sentence -> {sent}  {'OK' if good else 'FAIL'}")
        ok &= good
        if edited.decode([tid]) != new:
            print(f"  verify {new!r}: decode({tid}) = {edited.decode([tid])!r} FAIL")
            ok = False
    if not ok:
        raise SystemExit("verification failed; nothing written")

    # --- tokenizer_config.json / added_tokens.json ---
    cfg_path = os.path.join(args.model, "tokenizer_config.json")
    cfg = None
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf8") as f:
            cfg = json.load(f)
        atd = cfg.setdefault("added_tokens_decoder", {})
        for old, new, tid, _enc in plan:
            atd[str(tid)] = {"content": new, "lstrip": False, "normalized": False,
                             "rstrip": False, "single_word": False, "special": False}
    at_path = os.path.join(args.model, "added_tokens.json")
    at = None
    if os.path.exists(at_path):
        with open(at_path, encoding="utf8") as f:
            at = json.load(f)
        for old, new, tid, _enc in plan:
            at.pop(old, None)
            at[new] = tid

    if not args.write:
        print(f"dry run: {len(plan)} tokens would be renamed; pass --write with --out or --in-place")
        return

    out = args.model if args.in_place else args.out
    os.makedirs(out, exist_ok=True)
    def dump(path, obj):
        if args.in_place and os.path.exists(path):
            shutil.copy2(path, path + ".bak")
        with open(path, "w", encoding="utf8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"  wrote {path}")
    dump(os.path.join(out, "tokenizer.json"), tj)
    if cfg is not None:
        dump(os.path.join(out, "tokenizer_config.json"), cfg)
    if at is not None:
        dump(os.path.join(out, "added_tokens.json"), at)
    if not args.in_place:
        for fn in ("special_tokens_map.json", "chat_template.jinja", "chat_template.json",
                   "tokenizer.model", "generation_config.json", "config.json"):
            p = os.path.join(args.model, fn)
            if os.path.exists(p) and not os.path.exists(os.path.join(out, fn)):
                shutil.copy2(p, os.path.join(out, fn))
    print(f"done: {len(plan)} tokens repurposed. Train their rows with\n"
          f"  trainable_tokens: {[new for _o, new, _t, _e in plan]}")


if __name__ == "__main__":
    main()
