"""
Does the INFERENCE path reproduce the trainer's held-out loss?

Scores mala-data/semancy_test.jsonl through exllamav3's own forward (the path
qlora_infer_native.py generates from) for every arm of the v6/v7 comparison, and
compares against the numbers the trainer logged:

    base (step 0)   : 2.5230
    v7 ckpt-160     : 1.8098
    v7 ckpt-200     : 1.8517
    v6 best (@123)  : 1.8268
    v6 final (@160) : 1.8285

Loss definition is copied from the trainer's eval_loss(): shifted causal CE over
supervised positions only, meaned per example, then meaned over examples.

The base arm is the calibration check -- if "no adapter" does not land on ~2.523
then the harness (template, masking, dtype) disagrees with the trainer and NO
number here can be trusted.
"""
import argparse, json, os, shutil, sys, tempfile, time
import torch
import torch.nn.functional as F

os.chdir("/home/unstable/exl3/private/exl3-qlora")
sys.path.insert(0, "/home/unstable/exl3/private/exl3-qlora")

from exllamav3 import Config, Model, Tokenizer
from exllamav3.model.lora import LoRA
from training.qlora_train_native import build_sft_examples

MODEL = "/mnt/two/weights/Muse-Glimmer-30B-exl3-6.00bpw"
EVAL = "mala-data/semancy_test.jsonl"
CORRECT = 5.65685424949238          # alpha/sqrt(r) = 32/sqrt(32)
V7 = "out/muse_glimmer_semancy_v7"
V6 = "out/muse_glimmer_semancy_v6"

# (label, adapter_dir, module_lora_scale, per_linear_only, trainer_reference)
ARMS = [
    ("base, no adapter",                      None,                      None,    False, 2.5230),
    ("v7 ckpt-160  module_scale=1.0 (BUG)",   f"{V7}/checkpoint-00000160", 1.0,    False, None),
    ("v7 ckpt-160  module_scale=5.657 (FIX)", f"{V7}/checkpoint-00000160", CORRECT, False, 1.8098),
    ("v7 ckpt-160  per-linear only",          f"{V7}/checkpoint-00000160", None,   True,  None),
    ("v7 ckpt-200  module_scale=5.657",       f"{V7}/checkpoint-00000200", CORRECT, False, 1.8517),
    ("v6 final (step 160)",                   f"{V6}/final",              None,    False, 1.8285),
    ("v6 best  (step 123)",                   V6,                         None,    False, 1.8268),
]

ap = argparse.ArgumentParser()
ap.add_argument("--use-per-device", type=float, nargs="*", default=[22.0, 22.0])
ap.add_argument("--max-examples", type=int, default=0)
ap.add_argument("--ce-chunk", type=int, default=256)
ap.add_argument("--out", default=None)
args = ap.parse_args()

config = Config.from_directory(MODEL)
model = Model.from_config(config)
model.load(use_per_device=args.use_per_device, progressbar=True)
tokenizer = Tokenizer.from_config(config)
print(f" -- active devices {list(model.active_devices)}, output {model.output_device}")

examples = build_sft_examples(
    model, tokenizer, EVAL, args.max_examples, 3072,
    split="train", messages_key="messages",
    prompt_format="jinja",
    chat_template_file=os.path.join(MODEL, "chat_template.jinja"),
    clean_text=True, min_response_words=3,
)
print(f" -- {len(examples)} eval examples", flush=True)


@torch.inference_mode()
def held_out_loss():
    total, n = 0.0, 0
    for ex in examples:
        ids = torch.tensor([ex["input_ids"]], dtype=torch.long)
        lbl = torch.tensor([ex["labels"]], dtype=torch.long)
        logits = model.forward(ids)
        lg = logits[0, :-1]
        tgt = lbl[0, 1:].to(lg.device)
        idx = (tgt != -100).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        loss_sum = 0.0
        for i in range(0, idx.numel(), args.ce_chunk):
            sl = idx[i:i + args.ce_chunk]
            loss_sum += F.cross_entropy(lg[sl].float(), tgt[sl], reduction="sum").item()
        total += loss_sum / idx.numel()
        n += 1
        del logits, lg
    return total / n


def strip_to_per_linear(src):
    """Copy out only the per-linear adapter, leaving lora_modules.safetensors behind."""
    tmp = tempfile.mkdtemp()
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(os.path.join(src, f), os.path.join(tmp, f))
    return tmp


results = []
print("\n" + "=" * 86, flush=True)
for label, adapter, mscale, per_linear_only, ref in ARMS:
    t0 = time.time()
    tmp = None
    lora = None
    if adapter is not None:
        d = adapter
        if per_linear_only:
            tmp = d = strip_to_per_linear(adapter)
        lora = LoRA.from_directory(model, d, module_lora_scale=mscale)
    v = held_out_loss()
    if lora is not None:
        lora.unload()
    if tmp:
        shutil.rmtree(tmp)
    delta = f"{v - ref:+.4f}" if ref is not None else "   --  "
    reft = f"{ref:.4f}" if ref is not None else "  --  "
    print(f"{label:<40} {v:.4f}   trainer {reft}   diff {delta}   "
          f"[{time.time() - t0:.0f}s]", flush=True)
    results.append({"arm": label, "loss": v, "trainer_ref": ref})
print("=" * 86, flush=True)

if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
