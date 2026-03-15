#!/usr/bin/env python3
"""
Analyze diagnostic data from observation experiments.
Produces summary statistics and comparison across experiments.
"""

import json
import sys
import os
import math


def load_diagnostics(path):
    with open(path) as f:
        return json.load(f)


def summarize_experiment(name, diag):
    """Print summary of one experiment's diagnostics."""
    if not diag:
        print(f"\n{name}: NO DATA")
        return {}

    steps = [d['step'] for d in diag]
    rhos = [d.get('rho', 0) for d in diag]
    cos_phis = [d.get('cos_phi', 0) for d in diag]
    text_losses = [d.get('text_loss', 0) for d in diag]
    audio_losses = [d.get('audio_loss', 0) for d in diag]
    text_grad_norms = [d.get('text_grad_norm', 0) for d in diag]
    audio_grad_norms = [d.get('audio_grad_norm', 0) for d in diag]

    # Displacement
    displacements = [d.get('displacement', {}).get('total', 0) for d in diag]

    # Embedding rank
    text_ranks = [d.get('embedding_rank', {}).get('text_rank', 0) for d in diag]
    audio_ranks = [d.get('embedding_rank', {}).get('audio_rank', 0) for d in diag]

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")
    print(f"Steps: {steps[0]} → {steps[-1]} ({len(diag)} checkpoints)")
    print(f"\n--- Loss ---")
    print(f"  Text:  {text_losses[0]:.3f} → {text_losses[-1]:.3f}")
    print(f"  Audio: {audio_losses[0]:.3f} → {audio_losses[-1]:.3f}")

    print(f"\n--- D1: Gradient Norm Ratio ρ(t) ---")
    print(f"  Early (first 3): {sum(rhos[:3])/max(len(rhos[:3]),1):.4f}")
    print(f"  Mid (middle 3):  {sum(rhos[len(rhos)//2-1:len(rhos)//2+2])/3:.4f}" if len(rhos) > 5 else "  N/A")
    print(f"  Late (last 3):   {sum(rhos[-3:])/max(len(rhos[-3:]),1):.4f}")
    print(f"  Range: [{min(rhos):.4f}, {max(rhos):.4f}]")

    # Check if ρ~1 regime exists
    rho_near_1 = [r for r in rhos if 0.3 < r < 3.0]
    if rho_near_1:
        print(f"  ρ~1 regime: {len(rho_near_1)}/{len(rhos)} checkpoints ({100*len(rho_near_1)/len(rhos):.0f}%)")

    print(f"\n--- D2: Gradient Interference cos φ(t) ---")
    print(f"  Mean: {sum(cos_phis)/len(cos_phis):.4f}")
    print(f"  Range: [{min(cos_phis):.4f}, {max(cos_phis):.4f}]")
    negative_cos = [c for c in cos_phis if c < -0.01]
    print(f"  Negative (conflict): {len(negative_cos)}/{len(cos_phis)} checkpoints")

    # Per-layer interference
    all_layer_cos = {}
    for d in diag:
        for k, v in d.get('layer_cos', {}).items():
            all_layer_cos.setdefault(k, []).append(v)
    if all_layer_cos:
        print(f"\n  Per-layer cos φ (mean over training):")
        for layer in sorted(all_layer_cos.keys(), key=lambda x: int(x)):
            vals = all_layer_cos[layer]
            mean_cos = sum(vals) / len(vals)
            neg_frac = sum(1 for v in vals if v < 0) / len(vals)
            if int(layer) % 4 == 0 or int(layer) == 23:
                print(f"    Layer {layer:>2}: cos={mean_cos:+.4f}, neg_frac={neg_frac:.0%}")

    print(f"\n--- D3: Gradient Norms ---")
    print(f"  Text:  {text_grad_norms[0]:.1f} → {text_grad_norms[-1]:.1f}")
    print(f"  Audio: {audio_grad_norms[0]:.1f} → {audio_grad_norms[-1]:.1f}")

    print(f"\n--- D5: Parameter Displacement ---")
    print(f"  Total: {displacements[0]:.1f} → {displacements[-1]:.1f}")

    print(f"\n--- D6: Embedding Effective Rank ---")
    print(f"  Text:  {text_ranks[0]:.0f} → {text_ranks[-1]:.0f} (Δ={text_ranks[-1]-text_ranks[0]:+.0f})")
    print(f"  Audio: {audio_ranks[0]:.0f} → {audio_ranks[-1]:.0f} (Δ={audio_ranks[-1]-audio_ranks[0]:+.0f})")

    # Per-codebook losses
    cb_first = diag[0].get('cb_losses', {})
    cb_last = diag[-1].get('cb_losses', {})
    if cb_first:
        print(f"\n--- D7: Per-Codebook Losses ---")
        for cb in sorted(cb_first.keys()):
            print(f"  {cb}: {cb_first[cb]:.3f} → {cb_last.get(cb, 0):.3f}")

    # Basin width
    basins = [d for d in diag if 'basin' in d]
    if basins:
        print(f"\n--- D4: Basin Width ---")
        for b in basins:
            basin = b['basin']
            print(f"  Step {b['step']}: baseline={basin['baseline_loss']:.3f}")
            for eps, data in sorted(basin['perturbations'].items()):
                print(f"    ε={eps}: degradation={data['degradation']:.3f}")

    return {
        'name': name,
        'final_text_loss': text_losses[-1],
        'final_audio_loss': audio_losses[-1],
        'final_rho': rhos[-1],
        'mean_cos_phi': sum(cos_phis) / len(cos_phis),
        'text_rank_delta': text_ranks[-1] - text_ranks[0],
        'audio_rank_delta': audio_ranks[-1] - audio_ranks[0],
    }


def compare_experiments(results):
    """Compare key metrics across experiments."""
    if len(results) < 2:
        return

    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    print(f"{'Experiment':<20} {'Text Loss':<12} {'Audio Loss':<12} {'ρ (final)':<12} {'cos φ (mean)':<12} {'Text Rank Δ':<12} {'Audio Rank Δ':<12}")
    print("-" * 92)
    for r in results:
        print(f"{r['name']:<20} {r['final_text_loss']:<12.3f} {r['final_audio_loss']:<12.3f} "
              f"{r['final_rho']:<12.4f} {r['mean_cos_phi']:<12.4f} "
              f"{r['text_rank_delta']:<+12.0f} {r['audio_rank_delta']:<+12.0f}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    experiments = []
    for name in sorted(os.listdir(results_dir)):
        diag_path = os.path.join(results_dir, name, "diagnostics.json")
        if os.path.exists(diag_path):
            experiments.append((name, diag_path))

    if not experiments:
        print(f"No diagnostics found in {results_dir}/")
        return

    all_results = []
    for name, path in experiments:
        diag = load_diagnostics(path)
        result = summarize_experiment(name, diag)
        if result:
            all_results.append(result)

    compare_experiments(all_results)


if __name__ == "__main__":
    main()
