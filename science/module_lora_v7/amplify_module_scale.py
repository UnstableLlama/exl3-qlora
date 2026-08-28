"""Does the embed/head LoRA point in a USEFUL direction, or is it done?

S63 established from the saved 8-bit Adam state that the module LoRAs were not
starved: their per-element Adam step (|m|/(sqrt(v)+eps), the quantity that
actually sets step size -- raw gradient magnitude does not, because Adam is
scale-invariant and a global clip cancels out of m/sqrt(v)) sits in the same
band as the per-linear adapters, and their displacement-vs-step exponent
(0.22-0.34) matches the per-linear control (0.21-0.25) exactly.

So there is no bug. What is still open is whether the adapter is UNDERTRAINED
(right direction, magnitude just small -- 240 steps at lr 1e-5 is a small
budget) or CONVERGED (this dataset has nothing more to teach at the vocabulary
level, and 0.0011 nats is the answer).

Amplifying the learned delta separates those without retraining. The trained
delta is scale * (a @ b); multiply `scale` and walk further along the exact
direction training chose:

  - loss keeps IMPROVING as scale grows  -> direction is right, magnitude is
    the limit -> undertrained -> a higher-lr rerun is justified.
  - loss DEGRADES immediately past 1x    -> the adapter already sits at its
    local optimum -> 0.0011 nats is real and this is finished.

Same loss definition, model load and eval set as check_module_scale.py; the
per-linear-only arm is the floor (S62 measured 1.8530) and 1x is the trained
point (1.8519).
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
TRAINED = 5.65685424949238           # alpha/sqrt(r) = 32/sqrt(32), what v7 trained at
CKPT = "out/muse_glimmer_semancy_v7/checkpoint-00000160"

# (label, module_lora_scale, per_linear_only). None scale + per_linear_only = the floor.
MULTS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
ARMS = [("per-linear only (module LoRA stripped)", None, True)]
ARMS += [(f"module scale x{m:<5g} = {TRAINED*m:8.3f}", TRAINED * m, False) for m in MULTS]

ap = argparse.ArgumentParser()
ap.add_argument("--use-per-device", type=float, nargs="*", default=[22.0, 22.0])
ap.add_argument("--max-examples", type=int, default=0)
ap.add_argument("--ce-chunk", type=int, default=256)
ap.add_argument("--out", default="out/muse_glimmer_v6_v7_comparison/amplify_results.json")
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
    tmp = tempfile.mkdtemp()
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(os.path.join(src, f), os.path.join(tmp, f))
    return tmp


results, floor = [], None
print("\n" + "=" * 86, flush=True)
for label, mscale, per_linear_only in ARMS:
    t0 = time.time()
    tmp = d = CKPT
    if per_linear_only:
        tmp = d = strip_to_per_linear(CKPT)
    lora = LoRA.from_directory(model, d, module_lora_scale=mscale)
    v = held_out_loss()
    lora.unload()
    if per_linear_only:
        shutil.rmtree(tmp)
        floor = v
    vs = f"{v - floor:+.4f}" if floor is not None else "   --  "
    print(f"{label:<40} {v:.4f}   vs stripped {vs}   [{time.time() - t0:.0f}s]",
          flush=True)
    results.append({"arm": label, "scale": mscale, "loss": v,
                    "vs_stripped": (v - floor) if floor is not None else None})
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
print("=" * 86, flush=True)
print("negative 'vs stripped' = the module LoRA helps at that amplification")
