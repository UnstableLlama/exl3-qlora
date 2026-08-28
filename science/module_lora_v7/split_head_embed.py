"""Which module LoRA earned the -0.0036 nats -- the head, or the embedding?

S64's amplification sweep moved BOTH together (best point x8, -0.0036 vs the
per-linear-only floor 1.85299). The spectrum analysis then found the two modules
are doing very different things:

  head:  spread spectrum (effective rank ~10/32), seen tokens move 2.59x more
         than unseen, |delta| ANTI-correlated with frequency (rho -0.24).
         Real, targeted vocabulary adaptation.
  embed: per-token norms dead flat (seen/unseen 1.00, rho +0.010, every
         frequency decile 0.0279-0.0281). Its token axis rides `embed_a`, the
         kaiming-init half that S63 measured at 7.8% live rows -- so the
         token->subspace assignment is still essentially random init and all the
         learning sits in the shared `b`.

Prediction from that: nearly all of the -0.0036 is the head. This measures it.

The loader keys off "lm_head.lora_a" / "embed_tokens.lora_a" independently
(exllamav3/model/lora.py:329,340), so a checkpoint carrying only one pair loads
exactly that module and nothing else.

Arms: floor, head-only x1/x8, embed-only x1/x8, and both x8 as a harness check
against the 1.84934 the earlier sweep measured. ~65 s/arm off one model load.
"""
import argparse, json, os, shutil, sys, tempfile, time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

os.chdir("/home/unstable/exl3/private/exl3-qlora")
sys.path.insert(0, "/home/unstable/exl3/private/exl3-qlora")

from exllamav3 import Config, Model, Tokenizer
from exllamav3.model.lora import LoRA
from training.qlora_train_native import build_sft_examples

MODEL = "/mnt/two/weights/Muse-Glimmer-30B-exl3-6.00bpw"
EVAL = "mala-data/semancy_test.jsonl"
TRAINED = 5.65685424949238
CKPT = "out/muse_glimmer_semancy_v7/checkpoint-00000160"

ap = argparse.ArgumentParser()
ap.add_argument("--use-per-device", type=float, nargs="*", default=[22.0, 22.0])
ap.add_argument("--max-examples", type=int, default=0)
ap.add_argument("--ce-chunk", type=int, default=256)
ap.add_argument("--out", default="out/muse_glimmer_v6_v7_comparison/split_results.json")
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


_ml = safe_open(os.path.join(CKPT, "lora_modules.safetensors"), "pt")
MODULE_T = {k: _ml.get_tensor(k) for k in _ml.keys()}


def make_ckpt(keep):
    """Temp checkpoint with the per-linear adapter plus only the named modules.
    keep=() gives the per-linear-only floor (no lora_modules.safetensors at all)."""
    tmp = tempfile.mkdtemp()
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(os.path.join(CKPT, f), os.path.join(tmp, f))
    if keep:
        sub = {k: v for k, v in MODULE_T.items() if k.split(".")[0] in keep}
        save_file(sub, os.path.join(tmp, "lora_modules.safetensors"))
    return tmp


ARMS = [
    ("per-linear only (floor)",        (),                            None),
    ("head only   x1",                 ("lm_head",),                  1.0),
    ("head only   x8",                 ("lm_head",),                  8.0),
    ("embed only  x1",                 ("embed_tokens",),             1.0),
    ("embed only  x8",                 ("embed_tokens",),             8.0),
    ("both        x8  (harness check)", ("lm_head", "embed_tokens"),  8.0),
]

results, floor = [], None
print("\n" + "=" * 86, flush=True)
for label, keep, mult in ARMS:
    t0 = time.time()
    tmp = make_ckpt(keep)
    lora = LoRA.from_directory(
        model, tmp, module_lora_scale=(TRAINED * mult) if mult else None)
    v = held_out_loss()
    lora.unload()
    shutil.rmtree(tmp)
    if floor is None:
        floor = v
    vs = f"{v - floor:+.4f}" if label != ARMS[0][0] else "   --  "
    print(f"{label:<34} {v:.4f}   vs floor {vs}   [{time.time() - t0:.0f}s]", flush=True)
    results.append({"arm": label, "modules": list(keep), "mult": mult,
                    "loss": v, "vs_floor": v - floor})
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
print("=" * 86, flush=True)
print("negative 'vs floor' = that module helps at that amplification")
print("harness check should reproduce 1.84934 from amplify_results.json x8")
