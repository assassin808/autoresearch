#!/usr/bin/env python3
"""
Generate presentation slides v3.
Design: minimal text, data-driven charts, LaTeX math, professional layout.
Changes from v2:
  - Merged Finding 2 (GSNR/B_crit) + Finding 4 (Steps>>Batch) into one slide
  - New slide: Basin Width (eps=0.01 degradation over steps)
  - New slide: Gradient Dynamics (rho + GSNR evolution)
  - References slide with citations
"""

import os, json, math
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# ============================================================
# Design System
# ============================================================
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
    for i, (line_text, props) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line_text
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
    "29 Experiments  |  15 Diagnostics  |  30k+ Steps  |  A100 GPU",
    size=14, color=C_SUBTLE, align=PP_ALIGN.CENTER)

# ============================================================
# Slide 2: Background — Mini-Omni
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_NAVY)
txt(s, 0.8, 0.3, 11.7, 0.6, "Background: Mini-Omni Architecture & Training", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 5, C_NAVY, 3)

# Left: Architecture
rect(s, 0.5, 1.1, 5.8, 3.5, C_LIGHT_BG, C_BORDER)
multi(s, 0.8, 1.2, 5.3, 3.3, [
    ("Architecture", {'size': 18, 'bold': True, 'color': C_NAVY}),
    ("", {'size': 4}),
    ("Qwen2-0.5B backbone (~500M params)", {'size': 14}),
    ("+ Whisper encoder + learnable adapter", {'size': 14}),
    ("+ SNAC (multi-scale neural audio codec) 24kHz", {'size': 14}),
    ("", {'size': 4}),
    ("8 parallel streams per timestep:", {'size': 14, 'bold': True}),
    ("  Streams 0-6: audio codebooks (4160 vocab each)", {'size': 13, 'color': C_SUBTLE}),
    ("  Stream 7: text tokens (152K vocab)", {'size': 13, 'color': C_SUBTLE}),
    ("", {'size': 4}),
    ("Shared embedding: 181,120 rows", {'size': 14}),
    ("  Text rows: pretrained (Qwen2)", {'size': 13, 'color': C_GREEN}),
    ("  Audio rows: randomly initialized", {'size': 13, 'color': C_RED, 'bold': True}),
    ("  tie_word_embeddings=True (lm_head = wte)", {'size': 13, 'color': C_SUBTLE}),
])

# Right: Three-stage training
rect(s, 6.8, 1.1, 6.0, 3.5, C_LIGHT_BG, C_BORDER)
multi(s, 7.1, 1.2, 5.5, 3.3, [
    ("Three-Stage Training", {'size': 18, 'bold': True, 'color': C_NAVY}),
    ("", {'size': 4}),
    ("S1: Adapter Training", {'size': 15, 'bold': True, 'color': C_ACCENT}),
    ("  Train Whisper adapter only; freeze LLM backbone", {'size': 13}),
    ("  Goal: align audio encoder to LLM embedding space", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 4}),
    ("S2: Text Adaptation", {'size': 15, 'bold': True, 'color': C_GREEN}),
    ("  Train LLM backbone; freeze adapter", {'size': 13}),
    ("  Text-only tasks (T1T2, A1T2). No audio output.", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 4}),
    ("S3: Joint Training  <-- Our focus", {'size': 15, 'bold': True, 'color': C_RED}),
    ("  Unfreeze everything. 4 task types:", {'size': 13}),
    ("  T1T2 (text->text), T1A2 (text->audio)", {'size': 12, 'color': C_SUBTLE}),
    ("  A1T2 (audio->text), A1A2 (audio->audio)", {'size': 12, 'color': C_SUBTLE}),
])

# Bottom: Our setup
rect(s, 0.5, 4.9, 12.3, 1.5, RGBColor(0xe8, 0xea, 0xf6), C_NAVY)
multi(s, 0.8, 5.0, 11.7, 1.3, [
    ("Our Experimental Setup", {'size': 16, 'bold': True, 'color': C_NAVY}),
    ("Starting point: post-S1 checkpoint (Qwen2 pretrained + trained adapter + random audio embeddings)", {'size': 13}),
    ("Data: VoiceAssistant-400K (470K samples x 4 tasks = 1.88M sequences). Hardware: 1x A100 80GB (Narval).", {'size': 13, 'color': C_SUBTLE}),
    ("We investigate S3 training: why does audio loss plateau while text converges?", {'size': 14, 'bold': True, 'color': C_RED}),
])

# ============================================================
# Slide 3: The Problem
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_RED)
txt(s, 0.8, 0.3, 11.7, 0.6, "The Problem", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 2, C_RED, 3)

def plot_text_audio(buf, dpi):
    # Use real data from exp13 (direct S3 10k, post_s1 start)
    d13 = json.load(open('results/exp13_long_s3/diagnostics.json'))
    steps = [d['step'] for d in d13]
    text_loss = [d['text_loss'] for d in d13]
    audio_loss = [d['audio_loss'] for d in d13]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(steps, text_loss, '-', color='#1e88e5', linewidth=2, alpha=0.8, label='Text train loss')
    ax.plot(steps, audio_loss, '-', color='#e53e3e', linewidth=2, alpha=0.8, label='Audio train loss (sum 7 CB)')
    ax.axhline(y=58, color='gray', linestyle=':', alpha=0.3)
    ax.annotate('H_random = 58 (random guessing)', xy=(100, 59), fontsize=9, color='gray')
    ax.axhline(y=37, color='#e53e3e', linestyle='--', alpha=0.3)
    ax.annotate('plateau ~37', xy=(5000, 38), fontsize=9, color='#e53e3e')
    ax.set_xlabel('S3 Training Steps (from post-S1, no S2)', fontsize=11)
    ax.set_ylabel('Train Loss', fontsize=11)
    ax.set_title('Direct S3 Training: Text vs Audio (exp13, 10k steps, eff=32)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 65)
    ax.grid(alpha=0.2)
    ax.fill_between(steps, 33, 42, alpha=0.06, color='red')
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

buf = make_chart(plot_text_audio)
s.shapes.add_picture(buf, Inches(0.8), Inches(1.2), Inches(7), Inches(3.5))

rect(s, 8.2, 1.2, 4.5, 3.5, C_LIGHT_BG, C_BORDER)
multi(s, 8.5, 1.4, 4, 3.2, [
    ("At 3,000 S3 steps (train loss):", {'size': 14, 'bold': True, 'color': C_SUBTLE}),
    ("Text loss:  9 -> 1.44  (-77%)", {'size': 16, 'color': C_ACCENT, 'bold': True}),
    ("Audio loss: 58 -> 37   (-14%)", {'size': 16, 'color': C_RED, 'bold': True}),
    ("", {'size': 8}),
    ("GSNR (Gradient Signal-to-Noise Ratio):", {'size': 13, 'color': C_SUBTLE}),
    ("Audio GSNR = 0.08", {'size': 18, 'bold': True, 'color': C_RED}),
    ("Signal is 12.5x weaker than noise", {'size': 13, 'color': C_SUBTLE}),
    ("", {'size': 6}),
    ("cos(grad_text, grad_audio) = 0", {'size': 18, 'bold': True, 'color': C_PURPLE}),
    ("Not conflicting -- orthogonal", {'size': 13, 'color': C_SUBTLE}),
])

txt(s, 0.8, 5.2, 11.7, 1.0,
    "Question: Is this an optimization failure, gradient conflict, or fundamental capacity limit?",
    size=18, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)

# ============================================================
# Slide 3: The Verdict
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_GREEN)
txt(s, 0.8, 0.3, 11.7, 0.6, "The Verdict: Plateau Breaks -- Slowly", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 4, C_GREEN, 3)

def plot_val_trajectory(buf, dpi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Train loss (text + audio) for exp16 (S2->S3 30k)
    d16 = json.load(open('results/exp16_long_s2s3/diagnostics.json'))
    steps16 = [d['step'] for d in d16]
    text16 = [d['text_loss'] for d in d16]
    audio16 = [d['audio_loss'] for d in d16]

    ax1.plot(steps16, text16, '-', color='#1e88e5', linewidth=1.5, alpha=0.7, label='Text train loss')
    ax1.plot(steps16, audio16, '-', color='#e53e3e', linewidth=1.5, alpha=0.7, label='Audio train loss')
    ax1.set_xlabel('S3 Steps (from S2 checkpoint)', fontsize=11)
    ax1.set_ylabel('Train Loss', fontsize=11)
    ax1.set_title('Train Loss: S2->S3, 30k steps (exp16)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 55)
    ax1.grid(alpha=0.2)
    ax1.annotate('text ~1.5', xy=(25000, 3), fontsize=9, color='#1e88e5')
    ax1.annotate('audio ~25', xy=(25000, 27), fontsize=9, color='#e53e3e')

    # Right: Val A1A2 for exp16 + chain + exp13 (no S2)
    val16_steps = [d['step'] for d in d16 if 'val_task_losses' in d and d['step'] % 2000 == 0]
    val16 = [d['val_task_losses']['A1A2']['audio'] for d in d16 if 'val_task_losses' in d and d['step'] % 2000 == 0]

    # Chain combined
    steps_ch, val_ch = [], []
    for name, offset in [('omni_s3_r1', 0), ('omni_s3_r2', 5500), ('omni_s3_r3', 11000)]:
        try:
            dc = json.load(open(f'results/{name}/diagnostics.json'))
            for d in dc:
                vt = d.get('val_task_losses', {})
                if vt and 'A1A2' in vt and d['step'] % 1000 == 0:
                    steps_ch.append(offset + d['step'])
                    val_ch.append(vt['A1A2']['audio'])
        except: pass

    # exp13 (no S2)
    d13 = json.load(open('results/exp13_long_s3/diagnostics.json'))
    val13_steps = [d['step'] for d in d13 if 'val_task_losses' in d and d['step'] % 2000 == 0]
    val13 = [d['val_task_losses']['A1A2']['audio'] for d in d13 if 'val_task_losses' in d and d['step'] % 2000 == 0]

    ax2.plot(val16_steps, val16, 'o-', color='#1e88e5', linewidth=2, markersize=4, label='S2->S3 eff=32 (exp16)')
    ax2.plot(steps_ch, val_ch, 's-', color='#e53e3e', linewidth=2, markersize=4, label='S2->S3 eff=192 (chain)')
    ax2.plot(val13_steps, val13, '^-', color='#f57c00', linewidth=2, markersize=4, label='Direct S3 eff=32 (exp13)')
    ax2.axhline(y=49.1, color='gray', linestyle=':', alpha=0.4)
    ax2.set_xlabel('S3 Training Steps', fontsize=11)
    ax2.set_ylabel('Val A1A2 Audio Loss', fontsize=11)
    ax2.set_title('Val Audio Loss: Plateau Breaks with More Steps', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)
    ax2.annotate('32.3', xy=(15000, 33.5), fontsize=10, color='#e53e3e', fontweight='bold')
    ax2.annotate('36.8', xy=(28000, 37.8), fontsize=10, color='#1e88e5', fontweight='bold')
    ax2.annotate('44.9', xy=(8500, 46), fontsize=10, color='#f57c00')

    plt.tight_layout()
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
    ("Just 12x slower than text.", {'size': 14, 'color': C_SUBTLE}),
])

# ============================================================
# Slide 4: Gradient Orthogonality
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_PURPLE)
txt(s, 0.8, 0.3, 11.7, 0.6, "Finding 1: Orthogonal, Not Conflicting", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 3, C_PURPLE, 3)

def plot_cosphi(buf, dpi):
    fig, ax = plt.subplots(figsize=(5, 3))
    data = json.load(open('results/exp16_long_s2s3/diagnostics.json'))
    steps = [d['step'] for d in data if 'cos_phi' in d]
    phis = [d['cos_phi'] for d in data if 'cos_phi' in d]
    ax.scatter(steps, phis, s=15, alpha=0.6, color='#7b1fa2')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.fill_between([0, 31000], -0.05, 0.05, alpha=0.1, color='gray')
    ax.set_xlabel('Steps', fontsize=11)
    ax.set_ylabel('cos(grad_text, grad_audio)', fontsize=11)
    ax.set_title('D2: Gradient Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.15, 0.15)
    ax.grid(alpha=0.2)
    ax.annotate('mean = 0', xy=(15000, 0.06), fontsize=11, color='#7b1fa2', fontweight='bold')
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

buf = make_chart(plot_cosphi, 5, 3)
s.shapes.add_picture(buf, Inches(0.5), Inches(1.1), Inches(5.5), Inches(3.3))

rect(s, 6.5, 1.1, 6.3, 3.3, C_LIGHT_BG, C_BORDER)
multi(s, 6.8, 1.3, 5.8, 3.0, [
    ("D2: cos(phi) = 0.004 +/- 0.02", {'size': 18, 'bold': True, 'color': C_PURPLE}),
    ("Across ALL 29 experiments, 30k steps", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 10}),
    ("D15: Audio energy in text subspace = 0.1%", {'size': 16, 'bold': True, 'color': C_PURPLE}),
    ("(random baseline = 0.00005%)", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 10}),
    ("Why orthogonal?", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("Text emb: pretrained, structured directions", {'size': 13}),
    ("Audio emb: random N(0, 0.02), orthogonal directions", {'size': 13}),
    ("-> Backbone features align with text only", {'size': 13, 'color': C_RED}),
    ("-> Audio can't access text's learned features", {'size': 13, 'color': C_RED}),
])

rect(s, 0.5, 4.8, 12.3, 1.2, RGBColor(0xf3, 0xe5, 0xf5), C_PURPLE)
multi(s, 0.8, 4.9, 11.7, 1.0, [
    ("Implication: Text doesn't guide audio. They learn independently in orthogonal subspaces.", {'size': 16, 'bold': True, 'color': C_PURPLE}),
    ("PCGrad/gradient projection verified ineffective (exp3). Not a conflict problem.", {'size': 13, 'color': C_TEXT}),
])

# ============================================================
# Slide 5: GSNR + B_crit + Steps>>Batch (MERGED)
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_ACCENT)
txt(s, 0.8, 0.3, 11.7, 0.6, "Finding 2: GSNR (Gradient Signal-to-Noise Ratio) Theory", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 4, C_ACCENT, 3)

# Left top: Math formulas
def plot_math_merged(buf, dpi):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis('off')

    equations = [
        (0.02, 0.85, r'$\displaystyle \mathrm{GSNR}(B) = \frac{B \cdot \lVert G\rVert^2}{\mathrm{tr}(\Sigma)}, \quad B_{\mathrm{crit}} = \frac{\mathrm{tr}(\Sigma)}{\lVert G\rVert^2}$', 14),
        (0.02, 0.50, r'Total compute: $C(B) = S_{\min} \times (B + B_{\mathrm{crit}})$', 14),
        (0.02, 0.15, r'$\frac{dC}{dB} = S_{\min} > 0 \;\Rightarrow\; C$ \textbf{always increases with} $B$', 14),
    ]

    for x, y, eq, fs in equations:
        ax.text(x, y, eq, fontsize=fs, transform=ax.transAxes, verticalalignment='center')

    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    plt.rcParams['text.usetex'] = False

buf = make_chart(plot_math_merged, 7, 2.5)
s.shapes.add_picture(buf, Inches(0.3), Inches(1.0), Inches(5.8), Inches(2.0))

# Left bottom: C(B) curve to visually resolve the "contradiction"
def plot_cb_curve(buf, dpi):
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    B = np.arange(2, 200, 1)
    B_crit = 25
    C = B + B_crit  # proportional to S_min
    ax.plot(B, C, '-', color='#1e88e5', linewidth=2.5)
    ax.axvline(x=9, color='#2e7d32', linestyle='-', alpha=0.8, linewidth=2, label='B*=9 (optimal)')
    ax.axvline(x=25, color='#e53e3e', linestyle='--', alpha=0.6, linewidth=2, label='B_crit=25 (GSNR=1)')
    ax.axvline(x=32, color='#f57c00', linestyle=':', alpha=0.6, linewidth=2, label='eff=32 (ours)')
    ax.axvline(x=192, color='#7b1fa2', linestyle=':', alpha=0.6, linewidth=2, label='eff=192 (ours)')
    # Mark points
    ax.plot(9, 9+25, 'o', color='#2e7d32', markersize=10, zorder=5)
    ax.plot(25, 25+25, 's', color='#e53e3e', markersize=8, zorder=5)
    ax.plot(32, 32+25, '^', color='#f57c00', markersize=8, zorder=5)
    ax.annotate('C=34\n(optimal)', xy=(9, 34), xytext=(15, 20), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#2e7d32'), color='#2e7d32', fontweight='bold')
    ax.annotate('C=50', xy=(25, 50), xytext=(35, 42), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#e53e3e'), color='#e53e3e')
    ax.annotate('C=57', xy=(32, 57), xytext=(45, 52), fontsize=9, color='#f57c00')
    ax.set_xlabel('Batch Size (B)', fontsize=11)
    ax.set_ylabel('Total Compute C (samples)', fontsize=11)
    ax.set_title('C(B) = S_min x (B + B_crit):  always increasing!', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 130)
    ax.grid(alpha=0.2)
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

buf = make_chart(plot_cb_curve, 5.5, 2.8)
s.shapes.add_picture(buf, Inches(0.3), Inches(3.2), Inches(5.8), Inches(2.8))

# Right: explanation + table
rect(s, 6.5, 1.0, 6.3, 2.0, C_LIGHT_BG, C_BORDER)
multi(s, 6.8, 1.1, 5.8, 1.8, [
    ("Why more noisy steps beats fewer clean steps", {'size': 14, 'bold': True, 'color': C_NAVY}),
    ("", {'size': 4}),
    ("McCandlish et al. 2018: total compute C(B) = S_min x (B + B_crit)", {'size': 12, 'color': C_SUBTLE}),
    ("C always increases with B. Smaller B = fewer total samples needed.", {'size': 12}),
    ("", {'size': 4}),
    ("B_crit is NOT the optimal — it's just where GSNR=1 per step.", {'size': 13, 'bold': True, 'color': C_RED}),
    ("Below B_crit: each step is noisy, but you get MORE steps per sample.", {'size': 12}),
    ("Noise averages out over N steps (sqrt(N)), signal accumulates (N).", {'size': 12}),
])

# Right bottom: comparison table
rect(s, 6.5, 3.2, 6.3, 2.8, C_LIGHT_BG, C_BORDER)
multi(s, 6.8, 3.25, 5.8, 0.4, [
    ("Verified experimentally (same 960k total samples):", {'size': 13, 'bold': True, 'color': C_NAVY}),
])

cmp_data = [
    ["Config", "eff", "Steps", "GSNR/step", "Val A1A2"],
    ["exp16 (small batch)", "32", "30k", "0.64", "36.8"],
    ["exp15 (large batch)", "192", "5k", "3.84", "44.6"],
    ["exp17 (large, no S2)", "192", "5k", "3.84", "43.9"],
]
tbl(s, 6.8, 3.75, 5.8, 1.5, cmp_data)

multi(s, 6.8, 5.3, 5.8, 0.5, [
    ("Small batch: 2.8x fewer samples for same val.", {'size': 12, 'bold': True, 'color': C_GREEN}),
    ("Theory predicts 3.8x — consistent.", {'size': 11, 'color': C_SUBTLE}),
])

# ============================================================
# Slide 6: Two CKA Paths
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_ORANGE)
txt(s, 0.8, 0.3, 11.7, 0.6, "Finding 3: CKA (Centered Kernel Alignment) — Two Paths, Same Destination", size=26, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 3, C_ORANGE, 3)

def plot_cka(buf, dpi):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    d16 = json.load(open('results/exp16_long_s2s3/diagnostics.json'))
    steps16 = [d['step'] for d in d16 if 'cka' in d]
    cka16 = [d['cka']['layer12'] for d in d16 if 'cka' in d]
    d13 = json.load(open('results/exp13_long_s3/diagnostics.json'))
    steps13 = [d['step'] for d in d13 if 'cka' in d]
    cka13 = [d['cka']['layer12'] for d in d13 if 'cka' in d]

    ax.plot(steps16[:20], cka16[:20], '-', color='#1e88e5', linewidth=2.5, label='S2->S3 (CKA~0.99)')
    ax.plot(steps13, cka13, '-', color='#e53e3e', linewidth=2.5, label='Direct S3 (CKA->0.03)')
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
    ("CKA: measures representation similarity to init", {'size': 12, 'color': C_SUBTLE}),
    ("Same val, opposite CKA:", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("", {'size': 6}),
    ("Direct S3:  CKA=0.03  Val A1A2=44.9", {'size': 15, 'color': C_RED, 'font': 'Consolas'}),
    ("  -> Backbone completely reshaped", {'size': 13, 'color': C_SUBTLE}),
    ("  -> Audio gradients: output-first (layer23 -> layer0)", {'size': 13, 'color': C_SUBTLE}),
    ("", {'size': 6}),
    ("S2->S3:     CKA=0.99  Val A1A2=44.8", {'size': 15, 'color': C_ACCENT, 'font': 'Consolas'}),
    ("  -> Backbone nearly frozen", {'size': 13, 'color': C_SUBTLE}),
    ("  -> Audio gradients: input-first (layer0-2 as adapter)", {'size': 13, 'color': C_SUBTLE}),
    ("", {'size': 6}),
    ("-> Backbone is NOT the bottleneck", {'size': 16, 'bold': True, 'color': C_ORANGE}),
])

rect(s, 0.5, 5.0, 12.3, 1.2, RGBColor(0xff, 0xf3, 0xe0), C_ORANGE)
multi(s, 0.8, 5.1, 11.7, 1.0, [
    ("Audio bottleneck is in embedding space. Backbone is just a feature extractor -- reshape it or not, same result.", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("S2->S3 forms emergent LoRA: layer 0-2 become spontaneous audio adapters while deep layers stay frozen.", {'size': 13, 'color': C_TEXT}),
])

# ============================================================
# Slide 7: Basin Width (NEW)
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_GREEN)
txt(s, 0.8, 0.3, 11.7, 0.6, "Finding 4: Loss Landscape (Basin Width)", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 4, C_GREEN, 3)

def plot_basin(buf, dpi):
    fig, ax = plt.subplots(figsize=(8, 4))

    # exp16 S2->S3 30k
    d16 = json.load(open('results/exp16_long_s2s3/diagnostics.json'))
    steps16 = [d['step'] for d in d16 if 'basin' in d]
    basin16 = [d['basin']['perturbations']['0.01']['degradation'] for d in d16 if 'basin' in d]

    # exp13 Direct S3 10k
    d13 = json.load(open('results/exp13_long_s3/diagnostics.json'))
    steps13 = [d['step'] for d in d13 if 'basin' in d]
    basin13 = [d['basin']['perturbations']['0.01']['degradation'] for d in d13 if 'basin' in d]

    ax.plot(steps16, basin16, 'o-', color='#1e88e5', linewidth=2.5, markersize=5,
            label='S2->S3 30k (exp16)')
    ax.plot(steps13, basin13, 's-', color='#e53e3e', linewidth=2.5, markersize=5,
            label='Direct S3 10k (exp13)')

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Basin Width (eps=0.01 degradation)', fontsize=12)
    ax.set_title('Basin Width Evolution During Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)

    # Annotations
    ax.annotate(f'starts ~40', xy=(1000, 40), xytext=(3000, 20),
                fontsize=10, color='#1e88e5', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#1e88e5', lw=1.5))
    ax.annotate(f'starts ~336', xy=(1000, 336), xytext=(3000, 310),
                fontsize=10, color='#e53e3e', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e53e3e', lw=1.5))
    ax.annotate('widens with training', xy=(25000, 145), fontsize=10,
                color='#1e88e5', fontweight='bold')

    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

buf = make_chart(plot_basin, 8, 4)
s.shapes.add_picture(buf, Inches(0.3), Inches(1.1), Inches(7.5), Inches(4.0))

rect(s, 8.2, 1.1, 4.8, 4.0, C_LIGHT_BG, C_BORDER)
multi(s, 8.5, 1.3, 4.3, 3.7, [
    ("Basin Width (D3)", {'size': 16, 'bold': True, 'color': C_NAVY}),
    ("eps=0.01 perturbation", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 8}),
    ("S2->S3 (exp16):", {'size': 14, 'bold': True, 'color': C_ACCENT}),
    ("  Start: ~40 (flat basin)", {'size': 13, 'font': 'Consolas'}),
    ("  End:   ~163 (widens 4x)", {'size': 13, 'font': 'Consolas', 'color': C_GREEN}),
    ("", {'size': 6}),
    ("Direct S3 (exp13):", {'size': 14, 'bold': True, 'color': C_RED}),
    ("  Start: ~336 (sharp basin)", {'size': 13, 'font': 'Consolas'}),
    ("  End:   ~204 (narrows 1.6x)", {'size': 13, 'font': 'Consolas'}),
    ("", {'size': 8}),
    ("S2 pretraining gives a", {'size': 14}),
    ("flatter starting basin,", {'size': 14, 'bold': True, 'color': C_GREEN}),
    ("which widens further", {'size': 14, 'bold': True, 'color': C_GREEN}),
    ("with S3 training.", {'size': 14, 'bold': True, 'color': C_GREEN}),
])

rect(s, 0.5, 5.5, 12.3, 1.0, RGBColor(0xe8, 0xf5, 0xe9), C_GREEN)
multi(s, 0.8, 5.6, 11.7, 0.8, [
    ("S2 pretraining smooths the loss landscape (flatter basin), even though it doesn't improve final val loss.", {'size': 15, 'bold': True, 'color': C_NAVY}),
    ("This explains why S2->S3 training is more stable despite reaching the same accuracy as direct S3.", {'size': 13, 'color': C_TEXT}),
])

# ============================================================
# Slide 8: Gradient Dynamics (NEW)
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_RED)
txt(s, 0.8, 0.3, 11.7, 0.6, "Finding 5: Gradient Dynamics", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 4, C_RED, 3)

# Left chart: rho over steps
def plot_rho(buf, dpi):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    d16 = json.load(open('results/exp16_long_s2s3/diagnostics.json'))
    steps = [d['step'] for d in d16 if 'rho' in d]
    rho = [d['rho'] for d in d16 if 'rho' in d]

    # Smooth with rolling average for clarity
    window = 10
    rho_smooth = np.convolve(rho, np.ones(window)/window, mode='valid')
    steps_smooth = steps[window-1:]

    ax.scatter(steps, rho, s=8, alpha=0.2, color='#e53e3e')
    ax.plot(steps_smooth, rho_smooth, '-', color='#e53e3e', linewidth=2.5, label='rho (smoothed)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, label='rho=1 (balanced)')
    ax.set_xlabel('Steps', fontsize=11)
    ax.set_ylabel(r'$\rho$ = ||grad_audio|| / ||grad_text||', fontsize=11)
    ax.set_title(r'$\rho$: Gradient Magnitude Ratio', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.annotate('~0.6', xy=(500, 0.6), fontsize=10, color='#e53e3e', fontweight='bold')
    ax.annotate('~2.6', xy=(28000, 2.8), fontsize=10, color='#e53e3e', fontweight='bold')
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

buf = make_chart(plot_rho, 5, 3.2)
s.shapes.add_picture(buf, Inches(0.3), Inches(1.1), Inches(5.5), Inches(3.3))

# Right chart: GSNR evolution over R2+R3
def plot_gsnr_evo(buf, dpi):
    fig, ax = plt.subplots(figsize=(5, 3.2))

    d_r2 = json.load(open('results/omni_s3_r2/diagnostics.json'))
    d_r3 = json.load(open('results/omni_s3_r3/diagnostics.json'))

    # R2: global steps 5500-11000
    steps_r2 = [5500 + d['step'] for d in d_r2 if 'gsnr' in d]
    gsnr_r2 = [d['gsnr']['gsnr_audio'] for d in d_r2 if 'gsnr' in d]

    # R3: global steps 11000-16500
    steps_r3 = [11000 + d['step'] for d in d_r3 if 'gsnr' in d]
    gsnr_r3 = [d['gsnr']['gsnr_audio'] for d in d_r3 if 'gsnr' in d]

    all_steps = steps_r2 + steps_r3
    all_gsnr = gsnr_r2 + gsnr_r3

    ax.scatter(all_steps, all_gsnr, s=30, alpha=0.7, color='#f57c00', zorder=3)
    ax.axhline(y=0.08, color='gray', linestyle='--', alpha=0.5, label='mean ~0.08')

    # Trend line
    z = np.polyfit(all_steps, all_gsnr, 1)
    p = np.poly1d(z)
    ax.plot([all_steps[0], all_steps[-1]], [p(all_steps[0]), p(all_steps[-1])],
            '--', color='#f57c00', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Global Steps', fontsize=11)
    ax.set_ylabel('GSNR (audio)', fontsize=11)
    ax.set_title('D14: Audio GSNR over Chain R2+R3', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_ylim(0, 0.2)
    ax.annotate('flat ~0.08\n(no improvement)', xy=(13000, 0.10), fontsize=10,
                color='#f57c00', fontweight='bold')
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

buf = make_chart(plot_gsnr_evo, 5, 3.2)
s.shapes.add_picture(buf, Inches(6.3), Inches(1.1), Inches(5.5), Inches(3.3))

# Center text panel
rect(s, 0.5, 4.7, 12.3, 2.0, C_LIGHT_BG, C_BORDER)
multi(s, 0.8, 4.8, 11.7, 1.8, [
    ("Paradox: audio gradients get BIGGER (rho: 0.6 -> 2.6) but GSNR stays flat (~0.08)", {'size': 16, 'bold': True, 'color': C_RED}),
    ("", {'size': 4}),
    ("rho grows = audio loss gradient magnitude increases (model finds audio harder as text converges)", {'size': 13, 'color': C_TEXT}),
    ("GSNR flat = gradient DIRECTION stays noisy. Magnitude up, but signal/noise ratio unchanged.", {'size': 13, 'color': C_TEXT}),
    ("-> Learning rate scaling won't help: the problem is noise in direction, not magnitude.", {'size': 14, 'bold': True, 'color': C_NAVY}),
])

# ============================================================
# Slide 9: Linear Probe + Embedding
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_ACCENT)
txt(s, 0.8, 0.3, 11.7, 0.6, "Finding 6: Backbone Memorizes, Embeddings Learn Slowly", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 5, C_ACCENT, 3)

def plot_probe(buf, dpi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    layers = ['L6', 'L12', 'L18']
    train_acc = [18.6, 57.3, 77.1]
    val_acc = [6.2, 6.1, 6.2]
    x = np.arange(3)
    ax1.bar(x - 0.15, train_acc, 0.3, color='#1e88e5', label='Train', alpha=0.8)
    ax1.bar(x + 0.15, val_acc, 0.3, color='#e53e3e', label='Val', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Probe: Train vs Val (S2->S3 30k)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    for i, (tr, va) in enumerate(zip(train_acc, val_acc)):
        ax1.annotate(f'{int(tr/va)}x', xy=(i, tr+2), ha='center', fontsize=11, fontweight='bold', color='gray')
    ax1.grid(alpha=0.2, axis='y')

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
    ("-> backbone can't learn", {'size': 12, 'color': C_TEXT}),
    ("   stable audio rules", {'size': 12, 'color': C_TEXT}),
    ("", {'size': 4}),
    ("-> gradients noisy", {'size': 13, 'color': C_RED}),
    ("", {'size': 4}),
    ("-> embeddings learn", {'size': 13, 'color': C_GREEN}),
    ("   slowly (GSNR=0.08)", {'size': 12, 'color': C_TEXT}),
    ("", {'size': 4}),
    ("-> gradually structure", {'size': 13, 'color': C_GREEN}),
    ("   forms (cos: 0->0.12)", {'size': 12, 'color': C_TEXT}),
])

rect(s, 0.5, 5.3, 12.3, 0.8, RGBColor(0xe3, 0xf2, 0xfd), C_ACCENT)
txt(s, 0.8, 5.4, 11.7, 0.6,
    "D5 Bug Found: audio_emb_displacement reported 0 due to tie_word_embeddings. Actual displacement = 127 (via lm_head). Training IS correct.",
    size=13, color=C_TEXT)

# ============================================================
# Slide 10: Summary of Insights
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_ACCENT)
txt(s, 0.8, 0.3, 11.7, 0.6, "Key Insights", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 2, C_ACCENT, 3)

insights = [
    ("1", "Not Gradient Conflict", "cos(phi)=0, D15 energy=0.1%. Orthogonal subspaces, not interference.", C_PURPLE),
    ("2", "GSNR is Root Cause", "Audio GSNR=0.08 -> 12x more samples needed. More steps >> larger batch.", C_RED),
    ("3", "Plateau Breaks with Steps", "Val A1A2: 49->32 over 16.5k steps. Audio IS learning, slowly.", C_GREEN),
    ("4", "Basin Width Diverges", "S2->S3 starts flat (40) and widens 4x. Direct S3 starts sharp (336) and narrows.", C_ACCENT),
    ("5", "Backbone is NOT Bottleneck", "CKA (repr. similarity) 0.03 vs 0.99 -> same val. Bottleneck is embedding.", C_ORANGE),
    ("6", "Magnitude up, Direction flat", "Grad norm ratio grows 4x but GSNR (signal/noise) stays 0.08. Can't fix with LR.", C_NAVY),
]

y = 1.2
for num, title, desc, color in insights:
    rect(s, 0.5, y, 0.5, 0.45, color)
    txt(s, 0.55, y+0.02, 0.4, 0.4, num, size=20, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    txt(s, 1.2, y, 3.0, 0.45, title, size=16, bold=True, color=color)
    txt(s, 4.3, y+0.05, 8.5, 0.4, desc, size=14, color=C_TEXT)
    y += 0.85

# ============================================================
# Slide 11: Ongoing + Next Steps + References
# ============================================================
s = slide()
rect(s, 0, 0, W, 0.08, C_GREEN)
txt(s, 0.8, 0.3, 11.7, 0.6, "Ongoing & Next Steps", size=28, bold=True, color=C_TITLE)
line(s, 0.8, 0.85, 3, C_GREEN, 3)

rect(s, 0.5, 1.1, 5.8, 3.0, C_LIGHT_BG, C_BORDER)
multi(s, 0.8, 1.2, 5.3, 2.8, [
    ("Ongoing: Mini-Omni Chain", {'size': 18, 'bold': True, 'color': C_GREEN}),
    ("", {'size': 6}),
    ("R1-R3 complete (global 16,500 steps)", {'size': 14}),
    ("R4-R6 queued -> 27,500 steps (2.9 epochs)", {'size': 14}),
    ("eff=192, cosine LR over full schedule", {'size': 14}),
    ("", {'size': 6}),
    ("Audio trajectory:", {'size': 14, 'bold': True}),
    ("  R1: 29.9 -> R2: 25.6 -> R3: 23.9", {'size': 14, 'font': 'Consolas', 'color': C_GREEN}),
    ("  Val A1A2: 49 -> 35 -> 32", {'size': 14, 'font': 'Consolas', 'color': C_GREEN}),
    ("  Still declining, no convergence yet", {'size': 13, 'color': C_SUBTLE}),
])

rect(s, 6.8, 1.1, 6.0, 3.0, C_LIGHT_BG, C_BORDER)
multi(s, 7.1, 1.2, 5.5, 2.8, [
    ("Proposed Experiments", {'size': 18, 'bold': True, 'color': C_ORANGE}),
    ("", {'size': 6}),
    ("1. Optimal sampling (22% text, 78% audio)", {'size': 14}),
    ("   B_crit theory predicts efficiency gain", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 4}),
    ("2. LR scaling: eff=192 + LR x 7", {'size': 14}),
    ("   Should match small-batch efficiency", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 4}),
    ("3. Measure GSNR_text directly", {'size': 14}),
    ("   Currently estimated, need verification", {'size': 12, 'color': C_SUBTLE}),
    ("", {'size': 4}),
    ("4. 50k+ steps training", {'size': 14}),
    ("   Audio still improving, push further", {'size': 12, 'color': C_SUBTLE}),
])

rect(s, 0.5, 4.3, 12.3, 1.2, RGBColor(0xe8, 0xf5, 0xe9), C_GREEN)
multi(s, 0.8, 4.4, 11.7, 1.0, [
    ("Bottom Line", {'size': 18, 'bold': True, 'color': C_GREEN}),
    ("The audio plateau is a noise-dominated learning regime (GSNR=0.08), not an optimization or architecture failure.", {'size': 15, 'color': C_NAVY}),
    ("It breaks with sufficient training. The B_crit framework provides actionable guidance for compute-optimal training.", {'size': 14, 'color': C_TEXT}),
])

# References
rect(s, 0.5, 5.7, 12.3, 1.5, C_LIGHT_BG, C_BORDER)
multi(s, 0.8, 5.75, 11.7, 1.4, [
    ("References", {'size': 14, 'bold': True, 'color': C_NAVY}),
    ("[1] McCandlish et al. 2018, \"An Empirical Model of Large-Batch Training\" (B_crit theory)", {'size': 10, 'color': C_SUBTLE}),
    ("[2] Keskar et al. 2017, \"On Large-Batch Training for Deep Learning\" (sharp/flat minima)", {'size': 10, 'color': C_SUBTLE}),
    ("[3] Xie et al. 2024, \"Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming\"", {'size': 10, 'color': C_SUBTLE}),
    ("[4] Kornblith et al. 2019, \"Similarity of Neural Network Representations Revisited\" (CKA)", {'size': 10, 'color': C_SUBTLE}),
    ("[5] Chen et al. 2018, \"GradNorm: Gradient Normalization for Adaptive Loss Balancing\"", {'size': 10, 'color': C_SUBTLE}),
])

# ============================================================
# Save
# ============================================================
out = "slides_omni_basin_shaping.pptx"
prs.save(out)
print(f"Saved {out} ({len(prs.slides)} slides)")
