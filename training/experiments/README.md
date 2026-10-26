# QLoRA experiment tooling (not part of the training product)

Files here are one-off or experiment-specific tooling from the QLoRA-on-EXL3
sessions. They are kept for reproducibility of
past results but are NOT part of the reusable training path, and would not be
included in any upstream PR of the training work.

The reusable surface lives one level up in `training/`:

- `qlora_train_native.py` / `qlora_train_native_ddp.py` -- the trainers
- `qlora_train.py` + `qlora_train_config.yaml` -- the YAML launcher
- `qlora_validate_native.py` -- the correctness gates (run FIRST)
- `qlora_infer_native.py` -- before/after inference on the native path
- `qlora_train_bnb.py` -- the BNB-NF4 comparison arm (benchmark harness;
  experiment-side too, but it shares the run-log CSV schema with the native
  trainer, so it stays next to it)

In this directory:

- `make_style_dataset.py` -- generates dense style-transfer SFT sets (Yoda
  demo, Session 3); rewrites only the responses of a normal instruct set.
- `score_style_density.py` -- clause-inversion "Yoda-ness" metric used to
  score those runs (consumed by the trainers' `--gen-out`).
- `train_rocinante_yoda.sh` -- the overnight 2-GPU Rocinante-XL-16B run
  script (Session 3/4); paths and budgets are specific to that box/run, but
  the split-vs-ddp notes in its header are still a good worked example.

Moved here from `examples/` in the Session-14 tidy, then to
`training/experiments/` in the Session-19 repo cleanup, which moved all
training tooling out of `examples/` (restoring that directory to its upstream
state). Path references in the docs were rewritten to the new locations.
