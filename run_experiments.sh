#!/bin/bash
# Run all hypothesis experiments sequentially
# Each experiment modifies train.py flags, runs training, captures results

PYTHON=".venv/bin/python"
TRAIN="train.py"
RESULTS="results.tsv"

# Helper: set a flag value in train.py
set_flag() {
    local flag="$1"
    local value="$2"
    sed -i "s/^${flag} = .*/${flag} = ${value}/" "$TRAIN"
}

# Helper: run one experiment and log results
run_exp() {
    local desc="$1"
    echo ""
    echo "=========================================="
    echo "EXPERIMENT: $desc"
    echo "=========================================="

    # Commit the config change
    git add "$TRAIN"
    git commit -m "exp: $desc" --allow-empty 2>/dev/null
    COMMIT=$(git rev-parse --short HEAD)

    # Run training
    OUTPUT=$($PYTHON "$TRAIN" 2>&1)
    EXIT_CODE=$?
    echo "$OUTPUT" | tail -20

    # Parse results
    if [ $EXIT_CODE -ne 0 ] || echo "$OUTPUT" | grep -q "FAIL\|NaN\|nan"; then
        echo "${COMMIT}\t0.000000\t0.000000\t0.000000\t16.3\tcrash\t${desc}" >> "$RESULTS"
        echo ">>> CRASHED or FAILED"
        return 1
    fi

    VAL_LOSS=$(echo "$OUTPUT" | grep "^val_loss:" | awk '{print $2}')
    TEXT_BPB=$(echo "$OUTPUT" | grep "^text_bpb:" | awk '{print $2}')
    AUDIO_LOSS=$(echo "$OUTPUT" | grep "^audio_loss:" | awk '{print $2}')
    VRAM=$(echo "$OUTPUT" | grep "^peak_vram_mb:" | awk '{printf "%.1f", $2/1024}')

    if [ -z "$VAL_LOSS" ]; then
        echo "${COMMIT}\t0.000000\t0.000000\t0.000000\t16.3\tcrash\t${desc}" >> "$RESULTS"
        echo ">>> NO OUTPUT"
        return 1
    fi

    echo "${COMMIT}\t${VAL_LOSS}\t${TEXT_BPB}\t${AUDIO_LOSS}\t${VRAM}\tkeep\t${desc}" >> "$RESULTS"
    echo ">>> val_loss=${VAL_LOSS} text_bpb=${TEXT_BPB} audio_loss=${AUDIO_LOSS}"
    return 0
}

# Reset all flags to baseline
reset_flags() {
    set_flag "AUDIO_UPWEIGHT_SCHEDULE" "False"
    set_flag "GRAD_NORM_BALANCE" "False"
    set_flag "PERROW_LR" "False"
    set_flag "ADADECAY_WD" "False"
    set_flag "MODALITY_REBALANCE" "False"
    set_flag "DROPOUT" "0.05"
}

echo "Starting experiment sweep at $(date)"

# EXP 1: New baseline with more audio data
reset_flags
run_exp "new-baseline: 100h audio data, all flags off"

# EXP 2: H2 — Progressive audio upweighting
reset_flags
set_flag "AUDIO_UPWEIGHT_SCHEDULE" "True"
set_flag "AUDIO_UPWEIGHT_MAX" "1.5"
run_exp "H2: progressive audio upweight max=1.5"

# EXP 3: H2 variant — gentler upweighting
reset_flags
set_flag "AUDIO_UPWEIGHT_SCHEDULE" "True"
set_flag "AUDIO_UPWEIGHT_MAX" "1.2"
run_exp "H2: progressive audio upweight max=1.2"

# EXP 4: H3 — Gradient norm balancing
reset_flags
set_flag "GRAD_NORM_BALANCE" "True"
run_exp "H3: gradient norm balancing"

# EXP 5: H4 — Per-row adaptive LR
reset_flags
set_flag "PERROW_LR" "True"
set_flag "PERROW_LR_ALPHA" "0.3"
run_exp "H4: per-row LR alpha=0.3"

# EXP 6: H4 variant — stronger LR scaling
reset_flags
set_flag "PERROW_LR" "True"
set_flag "PERROW_LR_ALPHA" "0.5"
run_exp "H4: per-row LR alpha=0.5"

# EXP 7: H5 — AdaDecay WD
reset_flags
set_flag "ADADECAY_WD" "True"
set_flag "ADADECAY_BETA" "0.99"
run_exp "H5: AdaDecay WD beta=0.99"

# EXP 8: H5 variant — faster adaptation
reset_flags
set_flag "ADADECAY_WD" "True"
set_flag "ADADECAY_BETA" "0.95"
run_exp "H5: AdaDecay WD beta=0.95"

# EXP 9: H6 — MILES rebalancing
reset_flags
set_flag "MODALITY_REBALANCE" "True"
set_flag "MODALITY_REBALANCE_ALPHA" "0.5"
run_exp "H6: MILES rebalance alpha=0.5"

# EXP 10: H6 variant — stronger
reset_flags
set_flag "MODALITY_REBALANCE" "True"
set_flag "MODALITY_REBALANCE_ALPHA" "0.8"
run_exp "H6: MILES rebalance alpha=0.8"

# EXP 11: Best combo — combine H3+H5 if both help
reset_flags
set_flag "GRAD_NORM_BALANCE" "True"
set_flag "ADADECAY_WD" "True"
run_exp "combo: H3+H5 grad balance + AdaDecay"

# EXP 12: Re-tune per-row WD with new data
# With 100h audio (vs 5.2h), optimal audio WD should decrease
reset_flags
sed -i 's/^EMBED_WD_AUDIO = .*/EMBED_WD_AUDIO = 1.0/' "$TRAIN"
run_exp "per-row WD audio=1.0 (less WD for more data)"

# EXP 13: Even less audio WD
reset_flags
sed -i 's/^EMBED_WD_AUDIO = .*/EMBED_WD_AUDIO = 0.5/' "$TRAIN"
run_exp "per-row WD audio=0.5 (less WD for more data)"

# EXP 14: Moderate audio WD
reset_flags
sed -i 's/^EMBED_WD_AUDIO = .*/EMBED_WD_AUDIO = 1.5/' "$TRAIN"
run_exp "per-row WD audio=1.5"

# Reset to default audio WD
sed -i 's/^EMBED_WD_AUDIO = .*/EMBED_WD_AUDIO = 2.5/' "$TRAIN"

# Reset all flags
reset_flags
git add "$TRAIN"
git commit -m "reset flags after experiment sweep" 2>/dev/null

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE at $(date)"
echo "=========================================="
echo "Results in $RESULTS"
