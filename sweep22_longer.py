"""Sweep 22: Longer training (10 min) to test if optimizer improvements compound.

Key question: Does the gap between best and baseline GROW with more training steps?
If yes, the optimizer improvements are even more valuable at scale.

Also tests 10min baseline for comparison.
"""
import subprocess, re, os, json

PYTHON = ".venv/bin/python"

CONVERGENCE_LOGGING_CODE = '''
# Convergence logging
EVAL_INTERVAL_STEPS = 100
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
import json
with open("convergence_log.json", "w") as f:
    json.dump(convergence_log, f)
print(f"convergence_points: {len(convergence_log)}")
'''


def run_exp(desc, train_mods=None, prepare_mods=None, timeout=1200):
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
    modified_train = modified_train.replace(
        "    step += 1\n",
        LOGGING_INJECTION + "    step += 1\n",
        1
    )
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
        result = subprocess.run([PYTHON, "train.py"], capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
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


# Find TIME_BUDGET from prepare.py
import re as re2
with open("prepare.py") as f:
    m = re2.search(r'TIME_BUDGET\s*=\s*(\d+)', f.read())
    time_budget = int(m.group(1)) if m else 300

print(f"Base TIME_BUDGET = {time_budget}s")

experiments = [
    # 10min best
    ("long: full_best_10min", [],
     [(f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 2}")]),

    # 10min baseline
    ("long: baseline_10min", [
        ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
         "return WEIGHT_DECAY  # constant WD"),
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
        ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
        ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
        ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
    ], [(f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 2}")]),

    # 20min best (even longer)
    ("long: full_best_20min", [],
     [(f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 4}")]),

    # 20min baseline
    ("long: baseline_20min", [
        ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
         "return WEIGHT_DECAY  # constant WD"),
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
        ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
        ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
        ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
    ], [(f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 4}")]),
]

print(f"\nRunning {len(experiments)} longer experiments...")
results = []
timeouts = [1200, 1200, 2400, 2400]
for i, item in enumerate(experiments):
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=timeouts[i])
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 22 LONGER TRAINING SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
for desc, r in results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")

# Compare improvement at different training durations
print("\nIMPROVEMENT SCALING:")
pairs = [
    ("5min", results[0] if len(results) > 0 else None, results[1] if len(results) > 1 else None),
]
# 5min results from sweep21
five_min_best = 3.518  # from sweep21
five_min_base = 3.535
print(f"  5min:  best={five_min_best:.3f} base={five_min_base:.3f} Δ={five_min_best-five_min_base:.3f} ({100*(five_min_best-five_min_base)/five_min_base:.2f}%)")
if results[0][1] and results[1][1]:
    b, bl = results[0][1]['val_loss'], results[1][1]['val_loss']
    print(f"  10min: best={b:.3f} base={bl:.3f} Δ={b-bl:.3f} ({100*(b-bl)/bl:.2f}%)")
if len(results) > 3 and results[2][1] and results[3][1]:
    b, bl = results[2][1]['val_loss'], results[3][1]['val_loss']
    print(f"  20min: best={b:.3f} base={bl:.3f} Δ={b-bl:.3f} ({100*(b-bl)/bl:.2f}%)")
