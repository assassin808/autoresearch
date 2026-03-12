"""Sweep 21: Convergence curve logging for paper figures.

Logs val_loss at regular intervals during training to show HOW the
optimizer improvements affect training dynamics, not just final loss.

Key comparisons:
1. Full best vs original baseline — convergence trajectories
2. Longer training (10 min) — do improvements compound with more steps?
3. mom=0.97 + lower WD + coupledWD individual convergence curves

The convergence curves are the most important paper figure.
"""
import subprocess, re, os, json

PYTHON = ".venv/bin/python"

# We'll inject convergence logging into train.py
CONVERGENCE_LOGGING_CODE = '''
# Convergence logging: evaluate val loss every EVAL_INTERVAL_STEPS steps
EVAL_INTERVAL_STEPS = 50
convergence_log = []
'''

LOGGING_INJECTION = '''
    # Convergence checkpoint logging
    if step > 10 and step % EVAL_INTERVAL_STEPS == 0:
        model.eval()
        with torch.no_grad():
            with autocast_ctx:
                _vl, _tb, _al = evaluate_val_loss(model, tokenizer, DEVICE_BATCH_SIZE)
            convergence_log.append({
                "step": step,
                "time": total_training_time,
                "progress": progress,
                "val_loss": _vl,
                "text_bpb": _tb,
                "audio_loss": _al,
                "train_loss": debiased_smooth_loss,
            })
            print(f"  [EVAL step={step} t={total_training_time:.0f}s] val={_vl:.4f} text={_tb:.4f} audio={_al:.4f}")
        model.train()

'''

FINAL_LOG_INJECTION = '''
# Save convergence log
import json
with open("convergence_log.json", "w") as f:
    json.dump(convergence_log, f)
print(f"convergence_points: {len(convergence_log)}")
'''


def run_exp(desc, train_mods=None, prepare_mods=None):
    with open("train.py") as f:
        original_train = f.read()
    with open("prepare.py") as f:
        original_prepare = f.read()

    modified_train = original_train

    # Inject convergence logging
    modified_train = modified_train.replace(
        "train_loader = make_dataloader(",
        CONVERGENCE_LOGGING_CODE + "\ntrain_loader = make_dataloader(",
        1
    )
    # Inject eval in training loop (after the logging section)
    modified_train = modified_train.replace(
        "    step += 1\n",
        LOGGING_INJECTION + "    step += 1\n",
        1
    )
    # Inject final log save before val_loss output
    modified_train = modified_train.replace(
        "# Final summary",
        FINAL_LOG_INJECTION + "\n# Final summary",
        1
    )

    modified_prepare = original_prepare

    if train_mods:
        for old, new in train_mods:
            if modified_train.count(old) == 0:
                print(f"  WARNING: train.py not found: {old[:80]}...")
                with open("train.py", "w") as f:
                    f.write(original_train)
                with open("prepare.py", "w") as f:
                    f.write(original_prepare)
                return None
            modified_train = modified_train.replace(old, new, 1)
    if prepare_mods:
        for old, new in prepare_mods:
            if modified_prepare.count(old) == 0:
                print(f"  WARNING: prepare.py not found: {old[:80]}...")
                with open("train.py", "w") as f:
                    f.write(original_train)
                with open("prepare.py", "w") as f:
                    f.write(original_prepare)
                return None
            modified_prepare = modified_prepare.replace(old, new, 1)

    with open("train.py", "w") as f:
        f.write(modified_train)
    if prepare_mods:
        with open("prepare.py", "w") as f:
            f.write(modified_prepare)

    os.system(f'git add train.py prepare.py && git commit -m "exp: {desc}" --allow-empty 2>/dev/null')
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    print(f"\n{'='*60}\nEXP {commit}: {desc}\n{'='*60}")
    try:
        result = subprocess.run([PYTHON, "train.py"], capture_output=True, text=True, timeout=900)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        with open("results.tsv", "a") as f:
            f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f:
            f.write(original_train)
        with open("prepare.py", "w") as f:
            f.write(original_prepare)
        return None
    vl = re.search(r'^val_loss:\s+(\S+)', output, re.M)
    tb = re.search(r'^text_bpb:\s+(\S+)', output, re.M)
    al = re.search(r'^audio_loss:\s+(\S+)', output, re.M)
    if not vl or result.returncode != 0:
        print(f"  CRASHED")
        print(output[-1000:] if output else "no output")
        with open("results.tsv", "a") as f:
            f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f:
            f.write(original_train)
        with open("prepare.py", "w") as f:
            f.write(original_prepare)
        return None
    val_loss, text_bpb, audio_loss = float(vl.group(1)), float(tb.group(1)), float(al.group(1))
    print(f"  val_loss={val_loss:.6f} text_bpb={text_bpb:.6f} audio_loss={audio_loss:.6f}")
    with open("results.tsv", "a") as f:
        f.write(f"{commit}\t{val_loss:.6f}\t{text_bpb:.6f}\t{audio_loss:.6f}\t16.3\tkeep\t{desc}\n")

    # Save convergence log with experiment name
    if os.path.exists("convergence_log.json"):
        with open("convergence_log.json") as f:
            log = json.load(f)
        log_path = f"convergence_{desc.replace(' ', '_').replace(':', '_')}.json"
        with open(log_path, "w") as f:
            json.dump({"desc": desc, "commit": commit, "val_loss": val_loss,
                       "text_bpb": text_bpb, "audio_loss": audio_loss, "curve": log}, f)
        print(f"  Saved {len(log)} convergence points to {log_path}")
        os.remove("convergence_log.json")

    with open("train.py", "w") as f:
        f.write(original_train)
    with open("prepare.py", "w") as f:
        f.write(original_prepare)
    return {"val_loss": val_loss, "text_bpb": text_bpb, "audio_loss": audio_loss}


experiments = [
    # Standard 5-min runs with convergence logging
    ("conv: full_best_5min", [], None),

    ("conv: original_baseline_5min", [
        ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
         "return WEIGHT_DECAY  # constant WD"),
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
        ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
        ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
        ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
    ], None),

    # 10-min runs to test if improvements compound
    ("conv: full_best_10min", [
        ("TIME_BUDGET = 295", "TIME_BUDGET = 590"),
    ] if False else [], None),  # We'll enable after checking TIME_BUDGET

    # Only coupledWD (isolate its convergence effect)
    ("conv: only_coupledWD_5min", [
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
        ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
        ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
        ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
    ], None),

    # Only lower WD (isolate its convergence effect)
    ("conv: only_lowerWD_5min", [
        ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
         "return WEIGHT_DECAY  # constant WD"),
        ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
        ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
        ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
    ], None),

    # Only mom=0.97 (isolate its convergence effect)
    ("conv: only_mom97_5min", [
        ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
         "return WEIGHT_DECAY  # constant WD"),
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
        ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
    ], None),
]

# Check TIME_BUDGET in train.py
with open("train.py") as f:
    content = f.read()
    import re as re2
    tb_match = re2.search(r'TIME_BUDGET\s*=\s*(\d+)', content)
    if tb_match:
        time_budget = int(tb_match.group(1))
        print(f"TIME_BUDGET = {time_budget}s")
        # Add 10-min experiment with correct replacement
        experiments[2] = ("conv: full_best_10min", [
            (f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 2}"),
        ], None)
    else:
        print("WARNING: TIME_BUDGET not found, using prepare.py value")
        # TIME_BUDGET comes from prepare.py, need to check there
        with open("prepare.py") as pf:
            pcontent = pf.read()
            ptb = re2.search(r'TIME_BUDGET\s*=\s*(\d+)', pcontent)
            if ptb:
                time_budget = int(ptb.group(1))
                print(f"TIME_BUDGET (from prepare.py) = {time_budget}s")
                experiments[2] = ("conv: full_best_10min", [], [
                    (f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 2}"),
                ])

print(f"\nRunning {len(experiments)} convergence experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 21 CONVERGENCE SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")

# Load and compare convergence curves
print("\n" + "="*60)
print("CONVERGENCE CURVE COMPARISON")
print("="*60)
import glob
conv_files = sorted(glob.glob("convergence_conv_*.json"))
for cf in conv_files:
    with open(cf) as f:
        data = json.load(f)
    curve = data["curve"]
    if curve:
        print(f"\n{data['desc']} (final val_loss={data['val_loss']:.4f}):")
        print(f"  {'step':>6} | {'time':>6} | {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8}")
        for pt in curve[::2]:  # print every other point
            print(f"  {pt['step']:>6} | {pt['time']:>5.0f}s | {pt['val_loss']:.4f} | {pt['text_bpb']:.4f} | {pt['audio_loss']:.4f}")
