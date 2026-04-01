#!/usr/bin/env python3
"""
Generate professional presentation slides.
Design principles: minimal text, data-driven, math rendered via LaTeX.
"""

import os, json, math, subprocess, tempfile
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

# ============================================================
# Design System
# ============================================================
# Palette: professional dark navy + accent colors
C_BG       = RGBColor(0xff, 0xff, 0xff)
C_TITLE    = RGBColor(0x1a, 0x1a, 0x2e)
C_TEXT     = RGBColor(0x2d, 0x2d, 0x2d)
C_SUBTLE   = RGBColor(0x88, 0x88, 0x88)
C_ACCENT   = RGBColor(0x1e, 0x88, 0xe5)  # blue
C_RED      = RGBColor(0xe5, 0x3e, 0x3e)
C_GREEN    = RGBColor(0x2e, 0x7d, 0x32)
C_ORANGE   = RGBColor(0xf5, 0x7c, 0x00)
C_PURPLE   = RGBColor(0x7b, 0x1f, 0xa2)
C_NAVY     = RGBColor(0x0d, 0x27, 0x4d)
C_LIGHT_BG = RGBColor(0xf8, 0xf9, 0xfa)
C_BORDER   = RGBColor(0xde, 0xde, 0xde)
C_TABLE_H  = RGBColor(0x0d, 0x27, 0x4d)
C_WHITE    = RGBColor(0xff, 0xff, 0xff)

FONT = 'Calibri'
W = 13.333  # slide width inches
H = 7.5     # slide height inches

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)

# ============================================================
# Helper functions
# ============================================================
def slide():
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = C_BG
    return s

def txt(s, l, t, w, h, text, size=18, bold=False, color=C_TEXT, align=PP_ALIGN.LEFT, font=FONT):
    box = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font
    p.alignment = align
    return tf

def multi(s, l, t, w, h, lines, size=16, color=C_TEXT, spacing=8):
    """Multiple lines with controlled spacing."""
    box = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (line, props) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        sz = props.get('size', size)
        p.font.size = Pt(sz)
        p.font.bold = props.get('bold', False)
        p.font.color.rgb = props.get('color', color)
        p.font.name = props.get('font', FONT)
        p.space_after = Pt(props.get('space', spacing))
        p.alignment = props.get('align', PP_ALIGN.LEFT)
    return tf

def rect(s, l, t, w, h, color=C_LIGHT_BG, border=None):
    shape = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if border:
        shape.line.color.rgb = border
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def line(s, l, t, w, color=C_ACCENT, width=3):
    shape = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Pt(width))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()

def tbl(s, l, t, w, h, data, col_w=None):
    rows, cols = len(data), len(data[0])
    ts = s.shapes.add_table(rows, cols, Inches(l), Inches(t), Inches(w), Inches(h))
    table = ts.table
    if col_w:
        for i, cw in enumerate(col_w):
            table.columns[i].width = Inches(cw)
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(val)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.name = FONT
                p.alignment = PP_ALIGN.CENTER
                if r == 0:
                    p.font.bold = True
                    p.font.color.rgb = C_WHITE
                else:
                    p.font.color.rgb = C_TEXT
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_TABLE_H
            elif r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xf0, 0xf4, 0xf8)

def render_latex(tex, fontsize=16, dpi=200):
    """Render LaTeX to PNG image bytes."""
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.3 * tex.count('\\\\')))
    ax.axis('off')
    ax.text(0.5, 0.5, f'${tex}$', fontsize=fontsize, ha='center', va='center',
            transform=ax.transAxes, math_fontfamily='cm')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf

def add_latex(s, l, t, w, h, tex, fontsize=16):
    buf = render_latex(tex, fontsize)
    s.shapes.add_picture(buf, Inches(l), Inches(t), Inches(w), Inches(h))

def make_chart(fig_func, width=5, height=3, dpi=150):
    """Create chart from function, return BytesIO."""
    buf = BytesIO()
    fig_func(buf, dpi)
    buf.seek(0)
    return buf

# ============================================================
# Slide 1: Title
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.15, C_ACCENT)
txt(s, 1.5, 2.0, 10.3, 1.2,
    "Investigating the Audio Loss Plateau\nin Omni-Modal Joint Training",
    size=34, bold=True, color=C_TITLE, align=PP_ALIGN.CENTER)
line(s, 4, 3.7, 5.3, C_ACCENT, 3)
txt(s, 1.5, 4.0, 10.3, 0.6,
    "A Diagnostic Study with GSNR Analysis and Critical Batch Size Theory",
    size=18, color=C_SUBTLE, align=PP_ALIGN.CENTER)
txt(s, 1.5, 5.5, 10.3, 0.5,
    "29 Experiments  ·  15 Diagnostics  ·  30k+ Steps  ·  A100 GPU",
    size=14, color=C_SUBTLE, align=PP_ALIGN.CENTER)

# ============================================================
# Slide 2: The Problem (visual, minimal text)
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_RED)
txt(s, 0.8, 0.3, 11.7, 0.6, "The Problem", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 2, C_RED, 3)

# Left: text loss chart
def plot_text_audio(buf, dpi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    steps = [0, 500, 1000, 1500, 2000, 2500, 3000]
    text_loss = [11.9, 5.8, 3.4, 1.8, 1.6, 1.7, 1.4]
    audio_loss = [58, 52, 38, 31, 40, 37, 37]

    ax1.plot(steps, text_loss, 'o-', color='#1e88e5', linewidth=2.5, markersize=6, label='Text Loss')
    ax1.set_title('Text: Rapid Convergence', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Steps', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_ylim(0, 14)
    ax1.axhline(y=1.44, color='#1e88e5', linestyle='--', alpha=0.5)
    ax1.annotate('1.44 (-77%)', xy=(3000, 1.44), fontsize=10, color='#1e88e5')
    ax1.grid(alpha=0.2)

    ax2.plot(steps, audio_loss, 's-', color='#e53e3e', linewidth=2.5, markersize=6, label='Audio Loss')
    ax2.set_title('Audio: Plateau', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Steps', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_ylim(0, 65)
    ax2.axhline(y=37, color='#e53e3e', linestyle='--', alpha=0.5)
    ax2.axhline(y=58, color='gray', linestyle=':', alpha=0.3)
    ax2.annotate('37.25 (-14%)', xy=(2200, 38), fontsize=10, color='#e53e3e')
    ax2.annotate('H_random=58', xy=(100, 59), fontsize=9, color='gray')
    ax2.fill_between([500, 3000], 34, 42, alpha=0.1, color='red')
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

buf = make_chart(plot_text_audio)
s.shapes.add_picture(buf, Inches(0.8), Inches(1.2), Inches(7), Inches(3.5))

# Right: key numbers
rect(s, 8.2, 1.2, 4.5, 3.5, C_LIGHT_BG, C_BORDER)
multi(s, 8.5, 1.4, 4, 3.2, [
    ("At 3,000 steps:", {'size': 14, 'bold': True, 'color': C_SUBTLE}),
    ("Text loss:  9 → 1.44  (-77%)", {'size': 16, 'color': C_ACCENT, 'bold': True}),
    ("Audio loss: 58 → 37   (-14%)", {'size': 16, 'color': C_RED, 'bold': True}),
    ("", {'size': 8}),
    ("Audio GSNR = 0.08", {'size': 18, 'bold': True, 'color': C_RED}),
    ("Signal is 12.5× weaker than noise", {'size': 13, 'color': C_SUBTLE}),
    ("", {'size': 8}),
    ("cos(∇text, ∇audio) ≈ 0", {'size': 18, 'bold': True, 'color': C_PURPLE}),
    ("Not conflicting — orthogonal", {'size': 13, 'color': C_SUBTLE}),
])

txt(s, 0.8, 5.2, 11.7, 1.0,
    "Question: Is this an optimization failure, gradient conflict, or fundamental capacity limit?",
    size=18, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)

# ============================================================
# Slide 3: The Verdict (updated)
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_GREEN)
txt(s, 0.8, 0.3, 11.7, 0.6, "The Verdict: Plateau Breaks — Slowly", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 4, C_GREEN, 3)

# Chart: val A1A2 over steps
def plot_val_trajectory(buf, dpi):
    fig, ax = plt.subplots(figsize=(10, 4))
    # exp16 data
    steps16 = [2000, 4000, 6000, 8000, 10000, 14000, 18000, 22000, 26000, 30000]
    val16 = [49.5, 47.2, 45.5, 43.9, 42.5, 40.3, 38.8, 37.8, 37.2, 36.8]
    # chain data
    steps_ch = [1000, 3000, 5000, 7500, 10500, 13000, 16000]
    val_ch = [50.2, 45.8, 41.8, 38.0, 34.9, 33.4, 32.3]

    ax.plot(steps16, val16, 'o-', color='#1e88e5', linewidth=2, markersize=5,
            label='S2→S3 eff=32 (exp16)')
    ax.plot(steps_ch, val_ch, 's-', color='#e53e3e', linewidth=2, markersize=5,
            label='S2→S3 eff=192 (chain)')
    ax.axhline(y=49.1, color='gray', linestyle=':', alpha=0.4, label='Baseline (3k steps)')
    ax.set_xlabel('S3 Training Steps', fontsize=12)
    ax.set_ylabel('Val A1A2 Audio Loss', fontsize=12)
    ax.set_title('Audio Validation Loss Decreases with More Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    ax.annotate('49.1 (baseline)', xy=(500, 49.5), fontsize=9, color='gray')
    ax.annotate('36.8 (-25%)', xy=(28000, 37.5), fontsize=10, color='#1e88e5', fontweight='bold')
    ax.annotate('32.3 (-34%)', xy=(14500, 33), fontsize=10, color='#e53e3e', fontweight='bold')
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

buf = make_chart(plot_val_trajectory, 10, 4)
s.shapes.add_picture(buf, Inches(0.5), Inches(1.0), Inches(8), Inches(4.2))

rect(s, 8.8, 1.0, 4.2, 4.2, C_LIGHT_BG, C_BORDER)
multi(s, 9.0, 1.2, 3.8, 4.0, [
    ("Val A1A2 Trajectory", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("", {'size': 6}),
    ("3k steps:    49.1", {'size': 14, 'color': C_TEXT, 'font': 'Consolas'}),
    ("10k steps:   44.8  (-9%)", {'size': 14, 'color': C_TEXT, 'font': 'Consolas'}),
    ("16.5k steps: 32.3  (-34%)", {'size': 14, 'color': C_RED, 'font': 'Consolas', 'bold': True}),
    ("30k steps:   36.8  (-25%)", {'size': 14, 'color': C_ACCENT, 'font': 'Consolas'}),
    ("", {'size': 10}),
    ("CB0 Top-10 Accuracy", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("", {'size': 4}),
    ("3k:   20.7%", {'size': 14, 'font': 'Consolas'}),
    ("16.5k: 44.4%  (+114%)", {'size': 14, 'color': C_GREEN, 'font': 'Consolas', 'bold': True}),
    ("", {'size': 10}),
    ("Not permanent.", {'size': 16, 'bold': True, 'color': C_GREEN}),
    ("Just 12× slower than text.", {'size': 14, 'color': C_SUBTLE}),
])

# ============================================================
# Slide 4: Gradient Orthogonality
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_PURPLE)
txt(s, 0.8, 0.3, 11.7, 0.6, "Finding 1: Orthogonal, Not Conflicting", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 3, C_PURPLE, 3)

# cos phi chart
def plot_cosphi(buf, dpi):
    fig, ax = plt.subplots(figsize=(5, 3))
    data = json.load(open('results/exp16_long_s2s3/diagnostics.json'))
    steps = [d['step'] for d in data if 'cos_phi' in d]
    phis = [d['cos_phi'] for d in data if 'cos_phi' in d]
    ax.scatter(steps, phis, s=15, alpha=0.6, color='#7b1fa2')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.fill_between([0, 31000], -0.05, 0.05, alpha=0.1, color='gray')
    ax.set_xlabel('Steps', fontsize=11)
    ax.set_ylabel('cos(∇text, ∇audio)', fontsize=11)
    ax.set_title('D2: Gradient Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.15, 0.15)
    ax.grid(alpha=0.2)
    ax.annotate('mean ≈ 0', xy=(15000, 0.06), fontsize=11, color='#7b1fa2', fontweight='bold')
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

buf = make_chart(plot_cosphi, 5, 3)
s.shapes.add_picture(buf, Inches(0.5), Inches(1.1), Inches(5.5), Inches(3.3))

rect(s, 6.5, 1.1, 6.3, 3.3, C_LIGHT_BG, C_BORDER)
multi(s, 6.8, 1.3, 5.8, 3.0, [
    ("D2: cos(φ) = 0.004 ± 0.02", {'size': 18, 'bold': True, 'color': C_PURPLE}),
    ("Across ALL 29 experiments, 30k steps", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 10}),
    ("D15: Audio energy in text subspace = 0.1%", {'size': 16, 'bold': True, 'color': C_PURPLE}),
    ("(random baseline = 0.00005%)", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 10}),
    ("Why orthogonal?", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("Text emb: pretrained, structured directions", {'size': 13}),
    ("Audio emb: random N(0, 0.02), orthogonal directions", {'size': 13}),
    ("→ Backbone features align with text only", {'size': 13, 'color': C_RED}),
    ("→ Audio can't access text's learned features", {'size': 13, 'color': C_RED}),
])

# Bottom insight
rect(s, 0.5, 4.8, 12.3, 1.2, RGBColor(0xf3, 0xe5, 0xf5), C_PURPLE)
multi(s, 0.8, 4.9, 11.7, 1.0, [
    ("Implication: Text doesn't guide audio. They learn independently in orthogonal subspaces.", {'size': 16, 'bold': True, 'color': C_PURPLE}),
    ("PCGrad/gradient projection verified ineffective (exp3). Not a conflict problem.", {'size': 13, 'color': C_TEXT}),
])

# ============================================================
# Slide 5: GSNR + Critical Batch Size (MATH slide)
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_ACCENT)
txt(s, 0.8, 0.3, 11.7, 0.6, "Finding 2: GSNR Theory Explains Everything", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 4, C_ACCENT, 3)

# Math formulas via LaTeX
def plot_math(buf, dpi):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis('off')

    equations = [
        (0.02, 0.90, r'$\displaystyle \mathrm{GSNR}(B) = \frac{\lVert\mathbb{E}[\mathbf{g}]\rVert^2}{\mathrm{tr}(\Sigma)/B} = \frac{B \cdot \lVert G\rVert^2}{\mathrm{tr}(\Sigma)}$', 15),
        (0.02, 0.70, r'$\displaystyle B_{\mathrm{crit}} = \frac{\mathrm{tr}(\Sigma)}{\lVert G\rVert^2}$ \quad (noise $=$ signal threshold)', 15),
        (0.02, 0.48, r'Measured: $\mathrm{GSNR}_{\mathrm{audio}}(B{=}2) = 0.08 \;\Rightarrow\; B_{\mathrm{crit}}^{\mathrm{audio}} \approx 25$', 14),
        (0.20, 0.33, r'$\mathrm{GSNR}_{\mathrm{text}}(B{=}2) \approx 1.0 \;\Rightarrow\; B_{\mathrm{crit}}^{\mathrm{text}} \approx 2$', 14),
        (0.02, 0.12, r'Optimal: $B^* = B_{\mathrm{crit}}^T + \sqrt{B_{\mathrm{crit}}^T \cdot B_{\mathrm{crit}}^A} = 2 + \sqrt{50} \approx 9$', 15),
    ]

    for x, y, eq, fs in equations:
        ax.text(x, y, eq, fontsize=fs, transform=ax.transAxes, verticalalignment='center')

    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    plt.rcParams['text.usetex'] = False

buf = make_chart(plot_math, 12, 4.5)
s.shapes.add_picture(buf, Inches(0.3), Inches(1.0), Inches(8.5), Inches(3.8))

# Right: efficiency table
rect(s, 9.0, 1.0, 4.0, 4.5, C_LIGHT_BG, C_BORDER)
multi(s, 9.2, 1.1, 3.6, 0.5, [
    ("Sample Efficiency", {'size': 15, 'bold': True, 'color': C_NAVY}),
])

data = [
    ["B_eff", "GSNR", "Efficiency"],
    ["2", "0.08", "93%"],
    ["9 (B*)", "0.36", "100%"],
    ["25 (B_crit)", "1.00", "76%"],
    ["32 (ours)", "1.28", "70%"],
    ["192 (ours)", "7.68", "19%"],
]
tbl(s, 9.2, 1.7, 3.5, 2.5, data)

multi(s, 9.2, 4.3, 3.6, 1.0, [
    ("Our eff=192 wastes 81%", {'size': 13, 'bold': True, 'color': C_RED}),
    ("of compute vs optimal B*=9", {'size': 12, 'color': C_SUBTLE}),
])

# Bottom
rect(s, 0.5, 5.5, 12.3, 1.0, RGBColor(0xe3, 0xf2, 0xfd), C_ACCENT)
multi(s, 0.8, 5.6, 11.7, 0.8, [
    ("Key: Optimal B is NOT at B_crit. It's below — more noisy steps beats fewer clean steps.", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("Verified: exp16 (eff=32, 30k) uses 2.8× fewer samples than exp15 (eff=192, 5k) for same val.", {'size': 13, 'color': C_TEXT}),
])

# ============================================================
# Slide 6: Two CKA Paths
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_ORANGE)
txt(s, 0.8, 0.3, 11.7, 0.6, "Finding 3: Two Paths, Same Destination", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 3, C_ORANGE, 3)

# CKA chart
def plot_cka(buf, dpi):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    # S2->S3
    d16 = json.load(open('results/exp16_long_s2s3/diagnostics.json'))
    steps16 = [d['step'] for d in d16 if 'cka' in d]
    cka16 = [d['cka']['layer12'] for d in d16 if 'cka' in d]
    # Direct S3
    d13 = json.load(open('results/exp13_long_s3/diagnostics.json'))
    steps13 = [d['step'] for d in d13 if 'cka' in d]
    cka13 = [d['cka']['layer12'] for d in d13 if 'cka' in d]

    ax.plot(steps16[:20], cka16[:20], '-', color='#1e88e5', linewidth=2.5, label='S2→S3 (CKA≈0.99)')
    ax.plot(steps13, cka13, '-', color='#e53e3e', linewidth=2.5, label='Direct S3 (CKA→0.03)')
    ax.set_xlabel('Steps', fontsize=11)
    ax.set_ylabel('CKA @ Layer 12', fontsize=11)
    ax.set_title('Representation Drift from Init', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='center right')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

buf = make_chart(plot_cka, 5.5, 3.5)
s.shapes.add_picture(buf, Inches(0.3), Inches(1.1), Inches(6), Inches(3.5))

rect(s, 6.7, 1.1, 6.2, 3.5, C_LIGHT_BG, C_BORDER)
multi(s, 7.0, 1.3, 5.6, 3.2, [
    ("Same val performance, opposite CKA:", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("", {'size': 6}),
    ("Direct S3:  CKA=0.03  Val A1A2=44.9", {'size': 15, 'color': C_RED, 'font': 'Consolas'}),
    ("  → Backbone completely reshaped", {'size': 13, 'color': C_SUBTLE}),
    ("  → Audio gradients: output-first (layer23 → layer0)", {'size': 13, 'color': C_SUBTLE}),
    ("", {'size': 6}),
    ("S2→S3:     CKA=0.99  Val A1A2=44.8", {'size': 15, 'color': C_ACCENT, 'font': 'Consolas'}),
    ("  → Backbone nearly frozen", {'size': 13, 'color': C_SUBTLE}),
    ("  → Audio gradients: input-first (layer0-2 as adapter)", {'size': 13, 'color': C_SUBTLE}),
    ("", {'size': 6}),
    ("→ Backbone is NOT the bottleneck", {'size': 16, 'bold': True, 'color': C_ORANGE}),
])

rect(s, 0.5, 5.0, 12.3, 1.2, RGBColor(0xff, 0xf3, 0xe0), C_ORANGE)
multi(s, 0.8, 5.1, 11.7, 1.0, [
    ("Audio bottleneck is in embedding space. Backbone is just a feature extractor — reshape it or not, same result.", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("S2→S3 forms emergent LoRA: layer 0-2 become spontaneous audio adapters while deep layers stay frozen.", {'size': 13, 'color': C_TEXT}),
])

# ============================================================
# Slide 7: Steps >> Batch >> S2
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_GREEN)
txt(s, 0.8, 0.3, 11.7, 0.6, "Finding 4: More Steps Wins (at fixed compute)", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 4, C_GREEN, 3)

data = [
    ["Config", "S2?", "eff", "Steps", "Samples", "Val A1A2", "Δ"],
    ["Baseline", "No", "32", "3k", "96k", "49.1", "—"],
    ["exp12 S2→S3", "Yes", "32", "10k", "320k", "44.8", "-9%"],
    ["exp13 Direct S3", "No", "32", "10k", "320k", "44.9", "-9%"],
    ["exp15 S2→S3 big", "Yes", "192", "5k", "960k", "44.6", "-9%"],
    ["exp17 no S2 big", "No", "192", "5k", "960k", "43.9", "-11%"],
    ["exp16 S2→S3 long", "Yes", "32", "30k", "960k", "36.8", "-25%"],
    ["Chain R3", "Yes", "192", "16.5k", "3.2M", "32.3", "-34%"],
]
tbl(s, 0.5, 1.2, 12.3, 3.8, data, col_w=[2.5, 0.8, 0.8, 1.0, 1.2, 1.5, 1.0])

rect(s, 0.5, 5.3, 12.3, 1.5, C_LIGHT_BG, C_BORDER)
multi(s, 0.8, 5.4, 11.7, 1.3, [
    ("Three controlled comparisons (same 960k total samples):", {'size': 14, 'bold': True, 'color': C_NAVY}),
    ("  Steps:  exp16 (30k steps, eff=32) = 36.8  vs  exp15 (5k steps, eff=192) = 44.6    →  More steps wins", {'size': 13, 'font': 'Consolas'}),
    ("  S2:     exp12 (with S2) = 44.8  vs  exp13 (no S2) = 44.9                          →  S2 doesn't help val", {'size': 13, 'font': 'Consolas'}),
    ("  Batch:  exp17 (eff=192, no S2) = 43.9  vs  exp15 (eff=192, S2) = 44.6             →  S2 slightly hurts(!)", {'size': 13, 'font': 'Consolas'}),
])

# ============================================================
# Slide 8: Linear Probe + Embedding Learning
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_ACCENT)
txt(s, 0.8, 0.3, 11.7, 0.6, "Finding 5: Backbone Memorizes, Embeddings Learn Slowly", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 5, C_ACCENT, 3)

# probe chart
def plot_probe(buf, dpi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # Train vs val gap
    layers = ['L6', 'L12', 'L18']
    train_acc = [18.6, 57.3, 77.1]
    val_acc = [6.2, 6.1, 6.2]
    x = np.arange(3)
    ax1.bar(x - 0.15, train_acc, 0.3, color='#1e88e5', label='Train', alpha=0.8)
    ax1.bar(x + 0.15, val_acc, 0.3, color='#e53e3e', label='Val', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Probe: Train vs Val (S2→S3 30k)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    for i, (tr, va) in enumerate(zip(train_acc, val_acc)):
        ax1.annotate(f'{int(tr/va)}×', xy=(i, tr+2), ha='center', fontsize=11, fontweight='bold', color='gray')
    ax1.grid(alpha=0.2, axis='y')

    # CB0 top-k evolution
    steps = [3000, 5000, 10000, 16500, 20000, 30000]
    top1 = [4.8, 7.5, 11.1, 15.9, 13.4, 14.6]
    top10 = [20.7, 25.9, 33.7, 44.4, 40.7, 42.1]
    ax2.plot(steps, top10, 'o-', color='#2e7d32', linewidth=2, label='Top-10')
    ax2.plot(steps, top1, 's-', color='#f57c00', linewidth=2, label='Top-1')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('CB0 Audio Token Prediction', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.2)
    ax2.annotate('+114%', xy=(16000, 46), fontsize=10, color='#2e7d32', fontweight='bold')

    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

buf = make_chart(plot_probe, 10, 3.5)
s.shapes.add_picture(buf, Inches(0.3), Inches(1.1), Inches(9), Inches(3.8))

rect(s, 9.5, 1.1, 3.5, 3.8, C_LIGHT_BG, C_BORDER)
multi(s, 9.7, 1.3, 3.2, 3.5, [
    ("Chicken & Egg", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("", {'size': 6}),
    ("Embeddings random", {'size': 13, 'color': C_RED}),
    ("→ backbone can't learn", {'size': 12, 'color': C_TEXT}),
    ("   stable audio rules", {'size': 12, 'color': C_TEXT}),
    ("", {'size': 4}),
    ("→ gradients noisy", {'size': 13, 'color': C_RED}),
    ("", {'size': 4}),
    ("→ embeddings learn", {'size': 13, 'color': C_GREEN}),
    ("   slowly (GSNR=0.08)", {'size': 12, 'color': C_TEXT}),
    ("", {'size': 4}),
    ("→ gradually structure", {'size': 13, 'color': C_GREEN}),
    ("   forms (cos: 0→0.12)", {'size': 12, 'color': C_TEXT}),
])

rect(s, 0.5, 5.3, 12.3, 0.8, RGBColor(0xe3, 0xf2, 0xfd), C_ACCENT)
txt(s, 0.8, 5.4, 11.7, 0.6,
    "D5 Bug Found: audio_emb_displacement reported 0 due to tie_word_embeddings. Actual displacement = 127 (via lm_head). Training IS correct.",
    size=13, color=C_TEXT)

# ============================================================
# Slide 9: Joint B_crit Theory
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_NAVY)
txt(s, 0.8, 0.3, 11.7, 0.6, "Theory: Joint Optimization for Omni-Modal Training", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 5, C_NAVY, 3)

def plot_theory(buf, dpi):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    eqs = [
        (0.02, 0.88, r'$\displaystyle N_x(B_x) = S_{\min}^x \left(1 + \frac{B_{\mathrm{crit}}^x}{B_x}\right), \quad C = N \cdot B$', 15),
        (0.02, 0.68, r'Joint constraint (shared params): $N = \max\left(N_{\mathrm{text}},\, N_{\mathrm{audio}}\right)$', 14),
        (0.02, 0.50, r'With $\langle G_T, G_A \rangle = 0$: two independent problems sharing parameters', 14),
        (0.02, 0.30, r'Minimize $C(B,p)$: $p^* = B_{\mathrm{crit}}^T / B,\quad B^* = B_{\mathrm{crit}}^T + \sqrt{B_{\mathrm{crit}}^T \cdot B_{\mathrm{crit}}^A}$', 15),
        (0.02, 0.10, r'Result: $B^* = 2 + \sqrt{50} \approx 9,\; p^* = 22\%$ text, $B_A^* = 7 < B_{\mathrm{crit}}^A = 25$', 15),
    ]

    for x, y, eq, fs in eqs:
        ax.text(x, y, eq, fontsize=fs, transform=ax.transAxes, verticalalignment='center')

    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    plt.rcParams['text.usetex'] = False

buf = make_chart(plot_theory, 12, 4)
s.shapes.add_picture(buf, Inches(0.3), Inches(1.0), Inches(9.5), Inches(3.5))

rect(s, 10.0, 1.0, 3.0, 3.5, RGBColor(0xff, 0xf3, 0xe0), C_ORANGE)
multi(s, 10.2, 1.2, 2.6, 3.2, [
    ("Surprise:", {'size': 15, 'bold': True, 'color': C_ORANGE}),
    ("", {'size': 6}),
    ("At optimum,", {'size': 13}),
    ("audio is BELOW", {'size': 14, 'bold': True, 'color': C_RED}),
    ("its B_crit!", {'size': 14, 'bold': True, 'color': C_RED}),
    ("", {'size': 6}),
    ("GSNR_audio = 0.28", {'size': 13, 'font': 'Consolas'}),
    ("(noise > signal)", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 6}),
    ("More noisy steps", {'size': 13, 'bold': True}),
    ("beats fewer", {'size': 13, 'bold': True}),
    ("clean steps.", {'size': 13, 'bold': True}),
])

rect(s, 0.5, 4.8, 12.3, 1.2, RGBColor(0xe8, 0xea, 0xf6), C_NAVY)
multi(s, 0.8, 4.9, 11.7, 1.0, [
    ("Omni-modal training dilemma: shared parameters can't satisfy both B_crit simultaneously.", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("Solution: set B_T=B_crit_T, let B_A < B_crit_A, maximize steps. Or: asymmetric sampling (22% text, 78% audio).", {'size': 13, 'color': C_TEXT}),
])

# ============================================================
# Slide 10: Summary of Insights
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_ACCENT)
txt(s, 0.8, 0.3, 11.7, 0.6, "Key Insights", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 2, C_ACCENT, 3)

insights = [
    ("1", "Not Gradient Conflict", "cos(φ)=0, D15 energy=0.1%. Orthogonal subspaces, not interference.", C_PURPLE),
    ("2", "GSNR is Root Cause", "Audio GSNR=0.08 → 12× more samples needed. Noise >> signal.", C_RED),
    ("3", "Plateau Breaks with Steps", "Val A1A2: 49→32 over 16.5k steps. Audio IS learning, slowly.", C_GREEN),
    ("4", "S2 Doesn't Help Capability", "Same val at matched steps. Smooths landscape only (basin 2.4×).", C_ORANGE),
    ("5", "Backbone ≠ Bottleneck", "CKA=0.03 vs 0.99 → same val. Bottleneck is embedding GSNR.", C_ACCENT),
    ("6", "Emergent Adapter", "S2→S3: layer 0-2 become spontaneous audio LoRA. Deep layers frozen.", C_NAVY),
]

y = 1.2
for num, title, desc, color in insights:
    rect(s, 0.5, y, 0.5, 0.45, color)
    txt(s, 0.55, y+0.02, 0.4, 0.4, num, size=20, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    txt(s, 1.2, y, 3.0, 0.45, title, size=16, bold=True, color=color)
    txt(s, 4.3, y+0.05, 8.5, 0.4, desc, size=14, color=C_TEXT)
    y += 0.85

# ============================================================
# Slide 11: What's Next
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_GREEN)
txt(s, 0.8, 0.3, 11.7, 0.6, "Ongoing & Next Steps", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 3, C_GREEN, 3)

rect(s, 0.5, 1.1, 5.8, 3.5, C_LIGHT_BG, C_BORDER)
multi(s, 0.8, 1.2, 5.3, 3.3, [
    ("Ongoing: Mini-Omni Chain", {'size': 18, 'bold': True, 'color': C_GREEN}),
    ("", {'size': 6}),
    ("R1-R3 complete (global 16,500 steps)", {'size': 14}),
    ("R4-R6 queued → 27,500 steps (2.9 epochs)", {'size': 14}),
    ("eff=192, cosine LR over full schedule", {'size': 14}),
    ("", {'size': 6}),
    ("Audio trajectory:", {'size': 14, 'bold': True}),
    ("  R1: 29.9 → R2: 25.6 → R3: 23.9", {'size': 14, 'font': 'Consolas', 'color': C_GREEN}),
    ("  Val A1A2: 49 → 35 → 32", {'size': 14, 'font': 'Consolas', 'color': C_GREEN}),
    ("  Still declining, no convergence yet", {'size': 13, 'color': C_SUBTLE}),
])

rect(s, 6.8, 1.1, 6.0, 3.5, C_LIGHT_BG, C_BORDER)
multi(s, 7.1, 1.2, 5.5, 3.3, [
    ("Proposed Experiments", {'size': 18, 'bold': True, 'color': C_ORANGE}),
    ("", {'size': 6}),
    ("1. Optimal sampling (22% text, 78% audio)", {'size': 14}),
    ("   B_crit theory predicts efficiency gain", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 4}),
    ("2. LR scaling: eff=192 + LR×7", {'size': 14}),
    ("   Should match small-batch efficiency", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 4}),
    ("3. Measure GSNR_text directly", {'size': 14}),
    ("   Currently estimated, need verification", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 4}),
    ("4. 50k+ steps training", {'size': 14}),
    ("   Audio still improving, push further", {'size': 12, 'color': C_SUBTLE}),
])

rect(s, 0.5, 5.0, 12.3, 1.5, RGBColor(0xe8, 0xf5, 0xe9), C_GREEN)
multi(s, 0.8, 5.1, 11.7, 1.3, [
    ("Bottom Line", {'size': 18, 'bold': True, 'color': C_GREEN}),
    ("The audio plateau is a noise-dominated learning regime (GSNR=0.08), not an optimization or architecture failure.", {'size': 15, 'color': C_NAVY}),
    ("It breaks with sufficient training. The B_crit framework provides actionable guidance for compute-optimal training.", {'size': 14, 'color': C_TEXT}),
])

# ============================================================
# Save
# ============================================================
out = "slides_omni_basin_shaping.pptx"
prs.save(out)
print(f"Saved {out} ({len(prs.slides)} slides)")
