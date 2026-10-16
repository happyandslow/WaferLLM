#!/usr/bin/env python3
"""Generate GQA Implementation Design slides."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Palette (light theme) ────────────────────────────────
BG        = RGBColor(0xFF, 0xFF, 0xFF)   # white
TITLE_CLR = RGBColor(0x1A, 0x1A, 0x2E)   # very dark navy
ACCENT    = RGBColor(0x1B, 0x5E, 0x8A)   # dark steel blue
ACCENT2   = RGBColor(0x0E, 0x7C, 0x6B)   # dark teal
BODY_CLR  = RGBColor(0x2D, 0x2D, 0x3D)   # dark charcoal
DIM_CLR   = RGBColor(0x6B, 0x72, 0x80)   # medium gray
RED_CLR   = RGBColor(0xC0, 0x30, 0x30)   # darker red
GREEN_CLR = RGBColor(0x1E, 0x8C, 0x45)   # darker green
ORANGE_CLR= RGBColor(0xB8, 0x73, 0x08)   # darker orange

# ── Card / box fills for light theme ────────────────────
CARD_FILL    = RGBColor(0xF4, 0xF6, 0xFA)   # very light gray-blue
CARD_FILL_ALT= RGBColor(0xEE, 0xF1, 0xF6)   # slightly darker alt
HDR_FILL     = RGBColor(0x1A, 0x1A, 0x2E)   # dark header row
HDR_FONT     = RGBColor(0xFF, 0xFF, 0xFF)   # white text on dark header
ROW_EVEN     = RGBColor(0xF7, 0xF8, 0xFB)   # table row even
ROW_ODD      = RGBColor(0xED, 0xEF, 0xF4)   # table row odd
BORDER_DIM   = RGBColor(0xC8, 0xCC, 0xD5)   # muted border for unchanged items

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

# ── Helpers ──────────────────────────────────────────────
def add_bg(slide):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = BG

def add_text_box(slide, left, top, width, height, text, font_size=16,
                 color=BODY_CLR, bold=False, alignment=PP_ALIGN.LEFT,
                 font_name="Consolas"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    txBox.text_frame.word_wrap = True
    p = txBox.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox

def add_rich_box(slide, left, top, width, height):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    txBox.text_frame.word_wrap = True
    return txBox.text_frame

def add_para(tf, text, size=15, color=BODY_CLR, bold=False, space_before=Pt(4),
             space_after=Pt(2), alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    if len(tf.paragraphs) == 1 and tf.paragraphs[0].text == "":
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.space_before = space_before
    p.space_after = space_after
    p.alignment = alignment
    return p

def add_run(para, text, size=15, color=BODY_CLR, bold=False, font_name="Calibri"):
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = font_name
    return run

def add_rect(slide, left, top, width, height, fill_color=None, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.background()
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    return shape

def add_table_slide(slide, left, top, width, rows, cols, col_widths,
                    headers, data, header_color=ACCENT, data_color=BODY_CLR):
    row_h = Inches(0.45)
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, Inches(rows * 0.45))
    table = table_shape.table
    for i, w in enumerate(col_widths):
        table.columns[i].width = w
    # Style header
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        for para in cell.text_frame.paragraphs:
            para.font.size = Pt(13)
            para.font.color.rgb = HDR_FONT
            para.font.bold = True
            para.font.name = "Calibri"
            para.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = HDR_FILL
    # Style data
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            for para in cell.text_frame.paragraphs:
                para.font.size = Pt(12)
                para.font.color.rgb = data_color
                para.font.name = "Calibri"
                para.alignment = PP_ALIGN.CENTER
            cell.fill.solid()
            cell.fill.fore_color.rgb = ROW_EVEN if i % 2 == 0 else ROW_ODD
    return table_shape


# ═══════════════════════════════════════════════════════════
# SLIDE 1 — Problem Statement & Baseline
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide)

# Title
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "GQA Decode on WSE-3 — Problem Statement", 32, TITLE_CLR, True, font_name="Calibri")

# Subtitle
add_text_box(slide, Inches(0.8), Inches(1.05), Inches(8), Inches(0.4),
             "WSE-3 decode kernel lacks multi-head attention support", 16, ACCENT, False, font_name="Calibri")

# Left column — Problem
tf = add_rich_box(slide, Inches(0.8), Inches(1.6), Inches(5.8), Inches(4.5))
add_para(tf, "Current State (Broken)", 18, ACCENT, True, space_before=Pt(0))
add_para(tf, "")
add_para(tf, "  \u2022  Score X-reduce sums across ALL P PEs", 14, BODY_CLR)
add_para(tf, "     \u2192 mixes partial scores from different heads", 13, RED_CLR)
add_para(tf, "")
add_para(tf, "  \u2022  n_heads, pes_p_head params exist but are dead code", 14, BODY_CLR)
add_para(tf, "")
add_para(tf, "  \u2022  K/V weights always [dim, dim] — no per-head shapes", 14, BODY_CLR)
add_para(tf, "")
add_para(tf, "  \u2022  comm library ignores head count entirely", 14, BODY_CLR)
add_para(tf, "")
add_para(tf, "Goal", 18, ACCENT2, True)
add_para(tf, "")
add_para(tf, "  Enable GQA (n_kv_heads < n_heads) with compact KV", 14, BODY_CLR)
add_para(tf, "  cache and zero runtime Q-alignment overhead", 14, BODY_CLR)

# Right column — PE Grid diagram
box = add_rect(slide, Inches(7.2), Inches(1.6), Inches(5.3), Inches(4.2),
               fill_color=CARD_FILL, line_color=ACCENT)
tf2 = add_rich_box(slide, Inches(7.4), Inches(1.8), Inches(5.0), Inches(4.0))
add_para(tf2, "PE Grid (P=8, n_heads=2)", 16, ACCENT, True, space_before=Pt(0))
add_para(tf2, "")
add_para(tf2, "  X-axis: partitions dim    Y-axis: partitions seq_len", 13, DIM_CLR, font_name="Consolas")
add_para(tf2, "")
add_para(tf2, "  X-axis: [px=0  px=1  px=2  px=3] [px=4  px=5  px=6  px=7]", 13, BODY_CLR, font_name="Consolas")
add_para(tf2, "           \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500   \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500", 13, ACCENT, font_name="Consolas")
add_para(tf2, "                Head 0                    Head 1", 13, ACCENT2, font_name="Consolas")
add_para(tf2, "           4 PEs, X-reduce         4 PEs, X-reduce", 13, ACCENT2, font_name="Consolas")
add_para(tf2, "           within this block        within this block", 13, ACCENT2, font_name="Consolas")
add_para(tf2, "")
add_para(tf2, "  Correct reduce: scoped to pes_p_head PEs", 14, GREEN_CLR)
add_para(tf2, "  Current reduce: all P PEs \u2192 WRONG", 14, RED_CLR)

# Footer
add_text_box(slide, Inches(0.8), Inches(6.6), Inches(11), Inches(0.4),
             "WaferLLM \u2014 GQA Decode Design  \u2022  WSE-3-GQA", 11, DIM_CLR, font_name="Calibri")


# ═══════════════════════════════════════════════════════════
# SLIDE 2 — Three-Phase Plan
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "Three-Phase Implementation Plan", 32, TITLE_CLR, True, font_name="Calibri")

# Phase boxes
phases = [
    ("Phase 1: MHA  (n_heads > 1)", ACCENT,
     [("Head-scoped score reduce", BODY_CLR),
      ("New axis=2 route config: X_HEAD_2nd_RD_*, X_HEAD_BD", BODY_CLR),
      ("Limits X-reduce to pes_p_head = P/n_heads PEs", BODY_CLR),
      ("New fn: all_reduce_bsz_seq_len_head_scoped()", ACCENT2),
      ("Constraint: n_heads | group_num", ORANGE_CLR)]),
    ("Phase 2: GQA + Option C", ACCENT,
     [("Compact K/V cache: kv_dim_p_pe rows (not dim_p_pe)", BODY_CLR),
      ("Offline W_Q/W_O column permutation \u2192 interleaved Q", BODY_CLR),
      ("gqa_group_size GEMVs per PE for score & output", BODY_CLR),
      ("KV-head-scoped reduce: axis=3, pes_p_kv_head PEs", ACCENT2),
      ("Zero runtime Q-alignment overhead", GREEN_CLR)]),
    ("Phase 3: Incremental KV Cache", ACCENT,
     [("On-chip KV cache management (builds on Phase 2)", BODY_CLR),
      ("Host pre-loads prefilled cache before decode", BODY_CLR),
      ("process_kv() appends tokens; iter_num = prefill_len_p_pe", BODY_CLR),
      ("Variable-length score via @set_dsd_length", ACCENT2),
      ("No routing changes — inherits Phase 2 infrastructure", GREEN_CLR)]),
]

box_w = Inches(3.7)
box_h = Inches(3.8)
start_x = Inches(0.6)
gap = Inches(0.35)

for i, (title, title_clr, bullets) in enumerate(phases):
    left = start_x + i * (box_w + gap)
    top = Inches(1.5)
    add_rect(slide, left, top, box_w, box_h,
             fill_color=CARD_FILL, line_color=ACCENT)
    tf = add_rich_box(slide, left + Inches(0.2), top + Inches(0.15), box_w - Inches(0.4), box_h - Inches(0.3))
    add_para(tf, title, 16, title_clr, True, space_before=Pt(0))
    for bullet_text, bullet_clr in bullets:
        add_para(tf, "")
        add_para(tf, f"  \u2022  {bullet_text}", 12, bullet_clr)

# Arrow connectors between boxes
for i in range(2):
    x = start_x + (i + 1) * box_w + i * gap + Inches(0.05)
    tf = add_rich_box(slide, x, Inches(3.1), Inches(0.35), Inches(0.5))
    add_para(tf, "\u25B6", 24, ACCENT, True, space_before=Pt(0), alignment=PP_ALIGN.CENTER)

# Key constraint box
add_rect(slide, Inches(0.6), Inches(5.6), Inches(11.8), Inches(1.0),
         fill_color=CARD_FILL_ALT, line_color=ORANGE_CLR)
tf = add_rich_box(slide, Inches(0.9), Inches(5.7), Inches(11.2), Inches(0.8))
add_para(tf, "Key Constraints:  ", 14, ORANGE_CLR, True, space_before=Pt(0))
p = tf.paragraphs[0]
add_run(p, "n_heads | group_num    \u2022    dim % P == 0    \u2022    (n_kv_heads \u00d7 head_dim) % P == 0    \u2022    ", 13, BODY_CLR, font_name="Calibri")
add_run(p, "Degenerate: gqa_group_size=1 \u2192 identical to Phase 1", 13, DIM_CLR, font_name="Calibri")

add_text_box(slide, Inches(0.8), Inches(6.85), Inches(11), Inches(0.4),
             "WaferLLM \u2014 GQA Decode Design  \u2022  WSE-3-GQA", 11, DIM_CLR, font_name="Calibri")


# ═══════════════════════════════════════════════════════════
# SLIDE 3 — Q/K Alignment Problem & Option C
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "The Q/K Alignment Problem & Option C Solution", 32, TITLE_CLR, True, font_name="Calibri")

add_text_box(slide, Inches(0.8), Inches(1.05), Inches(10), Inches(0.4),
             "With compact KV cache (kv_dim_p_pe < dim_p_pe), Q and K\u1d40 on the same PE are misaligned", 15, RED_CLR, False, font_name="Calibri")

# Comparison table
headers = ["", "Option A: Replicate K\u1d40", "Option B: Runtime Q Scatter", "Option C: W_Q Permutation"]
data = [
    ["K\u1d40 cache/PE", "dim_p_pe \u00d7 seq (replicated)", "kv_dim_p_pe \u00d7 seq (compact)", "kv_dim_p_pe \u00d7 seq (compact)"],
    ["Q alignment", "None needed", "All-to-all within KV-head/step", "Offline W_Q/W_O permute (one-time)"],
    ["Runtime cost", "None", "(gqa-1)\u00b7kv_dim_p_pe f16/PE/step", "None"],
    ["LLaMA-70B mem", "256 KB/PE", "32 KB/PE", "32 KB/PE"],
    ["Verdict", "Simple but wasteful", "Dominated by Option C", "RECOMMENDED"],
]

col_w = [Inches(1.8), Inches(3.4), Inches(3.6), Inches(3.6)]
table_shape = add_table_slide(slide, Inches(0.4), Inches(1.6), Inches(12.4),
                               len(data) + 1, 4, col_w, headers, data)

# Highlight Option C column header
table = table_shape.table
cell = table.cell(0, 3)
for para in cell.text_frame.paragraphs:
    para.font.color.rgb = GREEN_CLR

# Highlight verdict row
for j in range(4):
    cell = table.cell(len(data), j)
    for para in cell.text_frame.paragraphs:
        if j == 3:
            para.font.color.rgb = GREEN_CLR
            para.font.bold = True
        elif j == 1:
            para.font.color.rgb = ORANGE_CLR
        elif j == 2:
            para.font.color.rgb = RED_CLR

# Option C explanation box
add_rect(slide, Inches(0.6), Inches(4.4), Inches(12.0), Inches(2.4),
         fill_color=CARD_FILL, line_color=ACCENT2)
tf = add_rich_box(slide, Inches(0.9), Inches(4.5), Inches(11.4), Inches(2.2))
add_para(tf, "Option C: How It Works", 17, ACCENT2, True, space_before=Pt(0))
add_para(tf, "")
add_para(tf, "  1.  Permute W_Q columns offline so each PE's Q is interleaved by GQA group:", 13, BODY_CLR)
add_para(tf, "       buf = [Q_g0(kv_dim_p_pe) | Q_g1(kv_dim_p_pe) | ... ]   \u2192 naturally aligned with local K\u1d40 rows", 12, ACCENT, font_name="Consolas")
add_para(tf, "")
add_para(tf, "  2.  Apply same permutation to W_O rows (preserves input/output contract)", 13, BODY_CLR)
add_para(tf, "")
add_para(tf, "  3.  Score GEMV: run gqa_group_size GEMVs per PE, each Kt = kv_dim_p_pe, with KV-head-scoped reduce", 13, BODY_CLR)
add_para(tf, "")
p = add_para(tf, "  ", 13, BODY_CLR)
add_run(p, "Result: ", 13, GREEN_CLR, True, font_name="Calibri")
add_run(p, "8\u00d7 memory savings (LLaMA-70B) with zero runtime overhead. Identity permutation when gqa_group_size=1.", 13, BODY_CLR, font_name="Calibri")

add_text_box(slide, Inches(0.8), Inches(6.85), Inches(11), Inches(0.4),
             "WaferLLM \u2014 GQA Decode Design  \u2022  WSE-3-GQA", 11, DIM_CLR, font_name="Calibri")


# ═══════════════════════════════════════════════════════════
# SLIDE 4 — Decode Pipeline Changes
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "Decode Pipeline — Phase 2 (GQA + Option C)", 32, TITLE_CLR, True, font_name="Calibri")

# Pipeline steps
steps = [
    ("rmsnorm_x()",                        "UNCHANGED",     DIM_CLR),
    ("xq_matvec_mult()  (W_Q permuted)",   "UNCHANGED",     DIM_CLR),
    ("xk/xv_matvec_mult()",                "Nt = kv_dim_p_pe",  ORANGE_CLR),
    ("QKV Y-reduce",                       "extent = bsz\u00d7(dim_p_pe + 2\u00d7kv_dim_p_pe)", ORANGE_CLR),
    ("xq_rope()",                          "loop gqa_group_size sub-blocks",  ORANGE_CLR),
    ("xk_rope()",                          "kv_dim_p_pe extent",  ORANGE_CLR),
    ("reconfig_allreduce_axis(3)",         "NEW: KV-head-scoped X",  ACCENT2),
    ("score_matvec_mult() \u00d7 gqa",     "Kt=kv_dim_p_pe, gqa GEMVs",  ACCENT2),
    ("softmax_score() \u00d7 gqa",         "loop gqa_group_size times",  ORANGE_CLR),
    ("output_matvec_mult() \u00d7 gqa",    "Vt=kv_dim_p_pe, gqa GEMVs",  ACCENT2),
    ("o_matvec_mult()  (W_O permuted)",    "UNCHANGED",     DIM_CLR),
    ("FFN pipeline",                       "ALL UNCHANGED", DIM_CLR),
]

# Two columns
col_count = 2
per_col = 6
for col in range(col_count):
    left = Inches(0.6) + col * Inches(6.2)
    for row_i in range(per_col):
        idx = col * per_col + row_i
        if idx >= len(steps):
            break
        name, status, clr = steps[idx]
        top = Inches(1.4) + row_i * Inches(0.72)

        # Step box
        is_changed = "UNCHANGED" not in status
        border_clr = clr if is_changed else BORDER_DIM
        add_rect(slide, left, top, Inches(5.8), Inches(0.62),
                 fill_color=CARD_FILL, line_color=border_clr)

        tf = add_rich_box(slide, left + Inches(0.15), top + Inches(0.05), Inches(5.5), Inches(0.52))
        p = add_para(tf, "", 13, BODY_CLR, space_before=Pt(0))
        add_run(p, name, 13, BODY_CLR, True, font_name="Consolas")
        add_run(p, f"   \u2192  {status}", 12, clr, False, font_name="Calibri")

# Legend
tf = add_rich_box(slide, Inches(0.8), Inches(5.9), Inches(11), Inches(1.0))
add_para(tf, "", 13, BODY_CLR, space_before=Pt(0))
p = tf.paragraphs[0]
add_run(p, "\u2588 Changed   ", 13, ORANGE_CLR, True, font_name="Calibri")
add_run(p, "\u2588 New/Critical   ", 13, ACCENT2, True, font_name="Calibri")
add_run(p, "\u2588 Unchanged        ", 13, DIM_CLR, True, font_name="Calibri")
add_run(p, "Degenerate check: gqa_group_size=1 \u2192 all loops run once, kv_dim_p_pe=dim_p_pe \u2192 identical to Phase 1", 12, DIM_CLR, font_name="Calibri")

add_text_box(slide, Inches(0.8), Inches(6.85), Inches(11), Inches(0.4),
             "WaferLLM \u2014 GQA Decode Design  \u2022  WSE-3-GQA", 11, DIM_CLR, font_name="Calibri")


# ═══════════════════════════════════════════════════════════
# SLIDE 5 — Memory Layouts & Constraints
# ═══════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text_box(slide, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "PE-Local Memory Layouts & Constraints", 32, TITLE_CLR, True, font_name="Calibri")

# Buffer layout table
headers2 = ["Buffer", "Shape (per PE)", "Dimension Order (slow \u2192 fast)", "Notes"]
data2 = [
    ["Q (interleaved)", "bsz \u00d7 dim_p_pe", "batch \u2192 GQA group \u2192 KV feature", "Option C interleaving"],
    ["K cache (K\u1d40)", "bsz \u00d7 kv_dim_p_pe \u00d7 max_seq_p_pe", "batch \u2192 KV feature \u2192 seq slot", "Stride-1 along seq axis"],
    ["V cache", "bsz \u00d7 max_seq_p_pe \u00d7 kv_dim_p_pe", "batch \u2192 seq slot \u2192 KV feature", "Contiguous token rows"],
    ["Score", "gqa \u00d7 bsz \u00d7 max_seq_p_pe", "batch \u2192 GQA group \u2192 seq slot", "iter_num-packed, not padded"],
]

col_w2 = [Inches(2.0), Inches(3.4), Inches(3.6), Inches(3.2)]
add_table_slide(slide, Inches(0.4), Inches(1.3), Inches(12.4),
                len(data2) + 1, 4, col_w2, headers2, data2)

# K vs V layout rationale
add_rect(slide, Inches(0.6), Inches(3.8), Inches(5.8), Inches(1.6),
         fill_color=CARD_FILL, line_color=ACCENT)
tf = add_rich_box(slide, Inches(0.8), Inches(3.9), Inches(5.4), Inches(1.4))
add_para(tf, "Why Different K/V Layouts?", 15, ACCENT, True, space_before=Pt(0))
add_para(tf, "")
add_para(tf, "  K\u1d40: feature outer / seq inner", 13, BODY_CLR, font_name="Consolas")
add_para(tf, "    \u2192 score GEMV walks over features", 12, DIM_CLR)
add_para(tf, "  V:  seq outer / feature inner", 13, BODY_CLR, font_name="Consolas")
add_para(tf, "    \u2192 output GEMV walks over tokens", 12, DIM_CLR)

# Constraints box
add_rect(slide, Inches(6.8), Inches(3.8), Inches(5.8), Inches(1.6),
         fill_color=CARD_FILL, line_color=ORANGE_CLR)
tf = add_rich_box(slide, Inches(7.0), Inches(3.9), Inches(5.4), Inches(1.4))
add_para(tf, "Constraints", 15, ORANGE_CLR, True, space_before=Pt(0))
add_para(tf, "")
add_para(tf, "  \u2022  dim % P == 0", 13, BODY_CLR, font_name="Consolas")
add_para(tf, "  \u2022  (n_kv_heads \u00d7 head_dim) % P == 0", 13, BODY_CLR, font_name="Consolas")
add_para(tf, "  \u2022  n_heads % n_kv_heads == 0", 13, BODY_CLR, font_name="Consolas")
add_para(tf, "  \u2022  n_heads | group_num", 13, BODY_CLR, font_name="Consolas")

# Phase 3 KV cache note
add_rect(slide, Inches(0.6), Inches(5.7), Inches(12.0), Inches(1.0),
         fill_color=CARD_FILL, line_color=ACCENT2)
tf = add_rich_box(slide, Inches(0.9), Inches(5.8), Inches(11.4), Inches(0.8))
add_para(tf, "Phase 3 KV Cache: ", 14, ACCENT2, True, space_before=Pt(0))
p = tf.paragraphs[0]
add_run(p, "Host pre-loads prefilled cache \u2192 iter_num = prefill_len_p_pe (not 0) \u2192 ", 13, BODY_CLR, font_name="Calibri")
add_run(p, "process_kv() appends from there \u2192 variable-length score via @set_dsd_length", 13, BODY_CLR, font_name="Calibri")
add_para(tf, "  Token distribution: PE(px,py) owns global tokens {py, py+P, py+2P, ...}. Batched GEMV uses nested bsz \u00d7 gqa_group_size loops.", 12, DIM_CLR)

add_text_box(slide, Inches(0.8), Inches(6.85), Inches(11), Inches(0.4),
             "WaferLLM \u2014 GQA Decode Design  \u2022  WSE-3-GQA", 11, DIM_CLR, font_name="Calibri")


# ── Save ─────────────────────────────────────────────────
out_path = "/mnt/raid0sata2/lexu/lexu/WaferLLM/Decode/WSE-3-GQA/docs/GQA_Decode_Design.pptx"
prs.save(out_path)
print(f"Saved to {out_path}")
