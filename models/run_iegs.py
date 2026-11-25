"""Run IncEventGS unmodified, or with a variant patched in at load time.

external/IncEventGS is never modified on disk: its main.py is read as text, the
variant's patches are applied in memory, and the result is executed as a module.
Each patch asserts how many times its anchor matches, so an upstream change fails
loudly instead of silently running the wrong experiment.

  python -m models.run_iegs --model inceventgs --config <yaml>
  python -m models.run_iegs --model edge_loss --beta 1 --config <yaml>
  add --random-init to skip the Marigold depth initialisation (IncEventGS-dagger in the paper)
"""
import argparse
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IEGS = REPO / "external" / "IncEventGS"


def _sub(src, pattern, repl, expected, flags=0):
    out, n = re.subn(pattern, repl, src, flags=flags)
    if n != expected:
        raise RuntimeError(f"patch anchor matched {n}x, expected {expected}: {pattern}")
    return out


def patch_edge_loss(src):
    src = _sub(src, r"(from loss_utils import [^\n]*\n)",
               r"\1from models.edge_loss import weighted_sq_err\n", 1)

    # initialization() and bundle_adjustment(), in file order: compute w after the GT event image
    stages = iter(['"init"', '"global_BA" if _global_BA else "BA"'])
    src = _sub(src,
               r"^( +)gt_events_acc = self\.post_process_event_image\(selected_event_stream, _sigma=blur_sigma\)[ \t]*$",
               lambda m: f"{m.group(0)}\n{m.group(1)}edge_w = self.edge_weight(selected_event_stream, {next(stages)})",
               2, re.M)
    src = _sub(src, r"\(event_mask\*\(gt_events_acc - syn_event_acc\)\*\*2\)",
               "(event_mask*weighted_sq_err((gt_events_acc - syn_event_acc)**2, edge_w))", 2)
    src = _sub(src, r"\(\(gt_events_acc - syn_event_acc\)\*\*2\)\.mean\(\)",
               "weighted_sq_err((gt_events_acc - syn_event_acc)**2, edge_w).mean()", 2)

    # tracking()
    src = _sub(src, r"^( +)loss_event = \(mask \* \(gt_events_acc - syn_event_acc\)\*\*2\)",
               r"\1edge_w = self.edge_weight(events_all[idx_ev_start:idx_ev_start+num_events_for_optim, :], 'tracking')\n"
               r"\1loss_event = (mask * weighted_sq_err((gt_events_acc - syn_event_acc)**2, edge_w))",
               1, re.M)
    return src


MODELS = {
    "inceventgs": [],                 # vanilla: nothing patched
    "edge_loss": [patch_edge_loss],   # E2EGS edge-weighted loss only
}


def load_iegs_main(patches):
    path = IEGS / "main.py"
    src = path.read_text(encoding="utf-8")
    for patch in patches:
        src = patch(src)
    mod = types.ModuleType("iegs_main")   # __name__ != "__main__", so its CLI block is skipped
    mod.__file__ = str(path)
    sys.modules["iegs_main"] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--config", required=True, help="an IncEventGS yaml")
    ap.add_argument("--output", help="default: work/runs/<config name>")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--random-init", action="store_true", help="depth_estimation_init = 0")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    out = Path(args.output).resolve() if args.output else REPO / "work" / "runs" / config_path.stem

    os.chdir(IEGS)                            # IncEventGS resolves inherit_from etc. from its own dir
    sys.path[:0] = [str(IEGS), str(REPO)]     # its modules first, then ours (models.*)

    import numpy                              # NumPy 2 dropped numpy.linalg.linalg (used in tools/eval_ate.py)
    if not hasattr(numpy.linalg, "linalg"):
        numpy.linalg.linalg = numpy.linalg

    mod = load_iegs_main(MODELS[args.model])
    cfg = mod.config.load_config(str(config_path))

    tag = args.model
    if args.model == "edge_loss":
        from models.edge_loss import EDGE_LOSS_DEFAULTS, edge_weight
        cfg["edge_loss"] = {**EDGE_LOSS_DEFAULTS, "beta": args.beta}
        mod.SLAM.edge_weight = edge_weight
        tag += f"_b{args.beta:g}"
    if args.random_init:
        cfg["depth_estimation_init"] = 0
        tag += "_rand"

    cfg["data"]["output"] = str(out)
    cfg["data"]["exp_name"] = tag
    save = out / tag
    for d in ["initialization", "tracking", "BA", "global_BA", "depth_for_new_gs", "img_eval/gt", "img_eval/est"]:
        (save / d).mkdir(parents=True, exist_ok=True)
    (save / "config.json").write_text(json.dumps(cfg, indent=4))
    shutil.copy(config_path, save / config_path.name)

    print(f"[run_iegs] model={args.model} tag={tag} -> {save}")
    mod.SLAM(cfg).run()


if __name__ == "__main__":
    main()
