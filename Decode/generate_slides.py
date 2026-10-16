#!/usr/bin/env python3
"""Generate GQA Implementation Comparison slides."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Color palette ──────────────────────────────────────────────────────────
BG_DARK   = RGBColor(0x1A, 0x1A, 0x2E)
BG_CARD   = RGBColor(0x24, 0x24, 0x3E)
ACCENT_1  = RGBColor(0x4E, 0xC9, 0xB0)  # Teal
ACCENT_2  = RGBColor(0x9C, 0xDC, 0xFE)  # Light blue
ACCENT_3  = RGBColor(0xCE, 0x91, 0x78)  # Salmon
ACCENT_4  = RGBColor(0xDC, 0xDC, 0xAA)  # Yellow-ish
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
GRAY      = RGBColor(0xB0, 0xB0, 0xB0)
LIGHT     = RGBColor(0xE0, 0xE0, 0xE0)
RED       = RGBColor(0xFF, 0x61, 0x61)
GREEN     = RGBColor(0x6A, 0xD6, 0x6A)
TABLE_HDR = RGBColor(0x2D, 0x2D, 0x50)
TABLE_ROW1= RGBColor(0x1E, 0x1E, 0x36)
TABLE_ROW2= RGBColor(0x26, 0x26, 0x42)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Helpers ────────────────────────────────────────────────────────────────

def set_bg(slide, color=BG_DARK):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_textbox(slide, left, top, width, height, text, font_size=18,
                color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return tf

def add_para(tf, text, font_size=18, color=WHITE, bold=False, space_before=Pt(4), space_after=Pt(2), indent=0, font_name="Calibri"):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.space_before = space_before
    p.space_after = space_after
    if indent:
        p.level = indent
    return p

def add_bullet_list(slide, left, top, width, height, items, font_size=16, color=WHITE, bullet_color=ACCENT_1):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_before = Pt(6)
        p.space_after = Pt(2)
    return tf

def add_card(slide, left, top, width, height, color=BG_CARD):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape

def add_table(slide, left, top, width, rows_data, col_widths, header=True):
    n_rows = len(rows_data)
    n_cols = len(rows_data[0])
    tbl_shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top),
                                        Inches(width), Inches(0.4 * n_rows))
    tbl = tbl_shape.table
    for ci, cw in enumerate(col_widths):
        tbl.columns[ci].width = Inches(cw)
    for ri, row in enumerate(rows_data):
        for ci, cell_text in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = cell_text
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(12)
                paragraph.font.name = "Calibri"
                if ri == 0 and header:
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = ACCENT_2
                else:
                    paragraph.font.color.rgb = LIGHT
                paragraph.alignment = PP_ALIGN.LEFT
            # Cell fill
            cf = cell.fill
            cf.solid()
            if ri == 0 and header:
                cf.fore_color.rgb = TABLE_HDR
            elif ri % 2 == 1:
                cf.fore_color.rgb = TABLE_ROW1
            else:
                cf.fore_color.rgb = TABLE_ROW2
    return tbl_shape

def add_code_block(slide, left, top, width, height, code, font_size=11):
    card = add_card(slide, left, top, width, height, RGBColor(0x1E, 0x1E, 0x30))
    tf = add_textbox(slide, left + 0.15, top + 0.1, width - 0.3, height - 0.2,
                     code, font_size=font_size, color=ACCENT_4, font_name="Consolas")
    tf.word_wrap = True
    return tf

def title_bar(slide, title, subtitle=None):
    # Accent line
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.06))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT_1
    shape.line.fill.background()
    # Title
    add_textbox(slide, 0.6, 0.25, 12, 0.7, title, font_size=32, color=WHITE, bold=True)
    if subtitle:
        add_textbox(slide, 0.6, 0.85, 12, 0.5, subtitle, font_size=16, color=GRAY)

def slide_number(slide, num, total):
    add_textbox(slide, 12.0, 7.0, 1.2, 0.4, f"{num} / {total}", font_size=11, color=GRAY, alignment=PP_ALIGN.RIGHT)

TOTAL = 15

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 1: Title
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])  # blank
set_bg(sl)
# Big accent bar
shape = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(2.8), Inches(13.333), Inches(0.08))
shape.fill.solid(); shape.fill.fore_color.rgb = ACCENT_1; shape.line.fill.background()
add_textbox(sl, 0.8, 1.2, 11.7, 1.4, "GQA Implementation Comparison",
            font_size=44, color=WHITE, bold=True)
add_textbox(sl, 0.8, 2.1, 11.7, 0.7, "WSE-3-GQA  vs  WSE-3-GQA-cheng",
            font_size=28, color=ACCENT_2)
add_textbox(sl, 0.8, 3.3, 11.7, 0.6, "Two approaches to Grouped Query Attention on Cerebras wafer-scale hardware",
            font_size=18, color=GRAY)
add_textbox(sl, 0.8, 5.5, 5, 0.5, "WaferLLM / Decode", font_size=16, color=GRAY)
add_textbox(sl, 0.8, 5.9, 5, 0.5, "March 2026", font_size=14, color=GRAY)
slide_number(sl, 1, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 2: Executive Summary
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "Executive Summary")

# Left card - GQA
add_card(sl, 0.5, 1.4, 5.9, 3.0, RGBColor(0x1C, 0x33, 0x2C))
add_textbox(sl, 0.7, 1.5, 5.5, 0.4, "WSE-3-GQA  (Original)", font_size=20, color=ACCENT_1, bold=True)
add_textbox(sl, 0.7, 1.95, 5.5, 0.35, "Dimensional GQA", font_size=15, color=ACCENT_4, bold=True)
add_bullet_list(sl, 0.7, 2.3, 5.5, 2.0, [
    "K/V dims reduced: kv_dim_p_pe < dim_p_pe",
    "Multi-group score: gqa_group_size loops",
    "KV-head-scoped reductions (axis=3)",
    "Full KV cache management + prefill",
    "Variable-length attention (iter_num)",
], font_size=13, color=LIGHT)

# Right card - cheng
add_card(sl, 6.9, 1.4, 5.9, 3.0, RGBColor(0x2C, 0x1C, 0x33))
add_textbox(sl, 7.1, 1.5, 5.5, 0.4, "WSE-3-GQA-cheng  (Alternative)", font_size=20, color=ACCENT_3, bold=True)
add_textbox(sl, 7.1, 1.95, 5.5, 0.35, "Replicated-Heads GQA", font_size=15, color=ACCENT_4, bold=True)
add_bullet_list(sl, 7.1, 2.3, 5.5, 2.0, [
    "All n_kv_heads stored per PE, runtime offset",
    "Single score GEMV (no multi-group)",
    "Standard reductions only (axis 0/1)",
    "No KV cache management",
    "Fixed-length attention, RoPE enabled",
], font_size=13, color=LIGHT)

# Bottom summary table
rows = [
    ["Aspect", "WSE-3-GQA", "WSE-3-GQA-cheng (full)"],
    ["GQA strategy", "Dimensional (asymmetric K/V dims)", "Replicated-heads (runtime offset)"],
    ["K/V weight per PE", "[dim_p_pe x kv_dim_p_pe]", "[n_kv_heads x dim_p_pe x dim_p_pe]"],
    ["KV cache mgmt", "Yes (process_kv, variable-len)", "No (pre-filled)"],
    ["Comm lib", "KV-head-scoped reductions (axis=3)", "Standard reductions only"],
    ["decode.csl / comm_pe.csl", "1282 / 1708 lines", "942 / 1086 lines"],
]
add_table(sl, 0.5, 4.6, 12.3, rows, [2.2, 5.0, 5.1])
slide_number(sl, 2, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 3: Shared Architecture
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "Shared Architecture", "Both implementations share the same decode pipeline")

add_card(sl, 0.5, 1.4, 12.3, 1.4)
add_textbox(sl, 0.7, 1.5, 12.0, 1.2,
    "RMSNorm(X) \u2192 QKV Proj \u2192 [RoPE] \u2192 [KV Cache] \u2192 Score \u2192 Softmax \u2192 Attn Output \u2192 O Proj \u2192 Residual \u2192 RMSNorm(Z) \u2192 UP+GATE \u2192 SiLU \u2192 Mul \u2192 DOWN \u2192 Residual",
    font_size=15, color=ACCENT_4, font_name="Consolas")

add_textbox(sl, 0.7, 3.1, 5.5, 0.4, "Common infrastructure", font_size=20, color=ACCENT_1, bold=True)
add_bullet_list(sl, 0.7, 3.5, 5.5, 3.5, [
    "P x P PE mesh with hierarchical 2-phase reduction",
    "Fused QKV all-reduce (all_reduce_bsz_dim_QKV_fusion)",
    "Fused UP+GATE all-reduce (all_reduce_bsz_ffn_dim_ZZ_fusion)",
    "Dynamic axis switching (X-axis / Y-axis reductions)",
    "Same fast_exp approx: (1 + x/256)^4",
    "Same GEMV primitive: @fmach mapped over left vector",
], font_size=14, color=LIGHT)

add_textbox(sl, 7.0, 3.1, 5.5, 0.4, "Where they diverge", font_size=20, color=ACCENT_3, bold=True)
add_bullet_list(sl, 7.0, 3.5, 5.5, 3.5, [
    "K/V weight dimensions and storage strategy",
    "QKV tile layout (asymmetric vs symmetric)",
    "Score computation (multi-group vs single)",
    "Communication scope (head-scoped vs global)",
    "KV cache management (present vs absent)",
    "Variable-length attention (yes vs no)",
], font_size=14, color=LIGHT)
slide_number(sl, 3, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 4: Core GQA Design — WSE-3-GQA (Dimensional)
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "Dimensional GQA  (WSE-3-GQA)", "GQA ratio encoded in K/V tensor dimensions at compile time")

add_textbox(sl, 0.7, 1.4, 12.0, 0.35, "Key formula:  kv_dim_p_pe = (n_kv_heads * head_dim) / P    |    gqa_group_size = dim_p_pe / kv_dim_p_pe",
            font_size=15, color=ACCENT_4, font_name="Consolas")

add_textbox(sl, 0.7, 1.9, 6, 0.4, "Llama-8B (P=256):  dim_p_pe=16, kv_dim_p_pe=4, groups=4",
            font_size=14, color=GRAY)

# Code block left
add_code_block(sl, 0.5, 2.4, 5.9, 2.3,
    "// Per-PE weight storage (minimal)\n"
    "Q_weight: [dim_p_pe x dim_p_pe]    // [16x16]\n"
    "K_weight: [dim_p_pe x kv_dim_p_pe] // [16x4]\n"
    "V_weight: [dim_p_pe x kv_dim_p_pe] // [16x4]\n\n"
    "// KV cache (reduced dim)\n"
    "XKCache: [bsz x kv_dim_p_pe x seq_len]\n"
    "XVCache: [bsz x seq_len x kv_dim_p_pe]",
    font_size=12)

# Explanation right
add_card(sl, 6.9, 2.4, 5.9, 2.3)
add_textbox(sl, 7.1, 2.5, 5.5, 0.35, "How GQA sharing works", font_size=16, color=ACCENT_1, bold=True)
add_bullet_list(sl, 7.1, 2.9, 5.5, 1.7, [
    "Q output (dim_p_pe) = gqa_group_size x kv_dim_p_pe",
    "Each group computes Q_group @ K^T independently",
    "All groups read the SAME K/V cache",
    "No runtime head mapping needed \u2014 sharing is structural",
    "Score tensor: [gqa_group_size x bsz x iter_num]",
], font_size=13, color=LIGHT)

# Bottom: score pseudocode
add_code_block(sl, 0.5, 5.0, 12.3, 2.0,
    "// score_matvec_mult() \u2014 multi-group with shared K cache\n"
    "for (b in bsz) {\n"
    "    for (g in gqa_group_size) {\n"
    "        score[g, b, :iter_num] = Q[b, g, :kv_dim_p_pe] @ K_cache[b, :, :]^T   // all groups share same K\n"
    "    }\n"
    "}\n"
    "// KV-head-scoped X-reduce: only within pes_p_kv_head PEs  (axis=3)\n"
    "all_reduce_bsz_g_var_seq_len_kv_head_scoped(...);\n"
    "score *= alpha;",
    font_size=12)
slide_number(sl, 4, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 5: Core GQA Design — WSE-3-GQA-cheng (Replicated)
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "Replicated-Heads GQA  (WSE-3-GQA-cheng)", "All KV heads stored per PE, selected at runtime via DSD offset")

add_textbox(sl, 0.7, 1.4, 12.0, 0.35,
    "Key formula:  kv_head_id = (px / pe_num_p_group) % n_kv_heads    |    offset = kv_head_id * dim_p_pe * dim_p_pe",
    font_size=15, color=ACCENT_4, font_name="Consolas")

# Code block left
add_code_block(sl, 0.5, 1.95, 5.9, 2.6,
    "// Per-PE weight storage (ALL KV heads)\n"
    "Q_weight: [dim_p_pe x dim_p_pe]           // [16x16]\n"
    "K_weight: [n_kv_heads x dim_p_pe x dim_p_pe]\n"
    "                                    // [8x16x16] = 2048\n"
    "V_weight: [n_kv_heads x dim_p_pe x dim_p_pe]\n\n"
    "// KV cache (ALL heads, full dim)\n"
    "XKCache: [n_kv_heads x dim_p_pe x seq_len]\n"
    "XVCache: [n_kv_heads x seq_len x dim_p_pe]",
    font_size=12)

# Explanation right
add_card(sl, 6.9, 1.95, 5.9, 2.6)
add_textbox(sl, 7.1, 2.05, 5.5, 0.35, "How GQA sharing works", font_size=16, color=ACCENT_3, bold=True)
add_bullet_list(sl, 7.1, 2.4, 5.5, 2.0, [
    "Every PE stores ALL n_kv_heads weight blocks",
    "At init: kv_head_id = quotient_x % n_kv_heads",
    "At runtime: DSD offset selects correct block",
    "vecmat_computation_dsd() snapshots offset per batch",
    "Only 1 of n_kv_heads blocks actually used per PE",
    "Standard global reductions (no head-scoped mode)",
], font_size=13, color=LIGHT)

# Bottom: K projection code
add_code_block(sl, 0.5, 4.8, 12.3, 2.2,
    "// xk_matvec_mult() \u2014 DSD offset selects KV head block\n"
    "fn xk_matvec_mult() void {\n"
    "    var k_offset: i16 = kv_head_id * dim_p_pe * dim_p_pe;\n"
    "    right_matrix_dsd = @set_dsd_base_addr(right_matrix_dsd, ptr_K_weight);\n"
    "    right_matrix_dsd = @increment_dsd_offset(right_matrix_dsd, k_offset, f16);\n"
    "    Kt = dim_p_pe;  Nt = dim_p_pe;  // Full dim \u2014 NOT reduced\n"
    "    vecmat_computation_dsd();  // Snapshot-based: preserves offset per batch row\n"
    "}",
    font_size=12)
slide_number(sl, 5, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 6: Memory Comparison
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "Per-PE Memory Comparison", "Llama-8B  |  P=256  |  n_heads=32  |  n_kv_heads=8  |  4:1 GQA ratio")

rows = [
    ["Buffer", "WSE-3-GQA", "WSE-3-GQA-cheng", "Ratio"],
    ["Q weight", "16 x 16 = 256 f16", "16 x 16 = 256 f16", "1x"],
    ["K weight", "16 x 4 = 64 f16", "8 x 16 x 16 = 2048 f16", "32x"],
    ["V weight", "16 x 4 = 64 f16", "8 x 16 x 16 = 2048 f16", "32x"],
    ["K cache (seq/PE=20)", "1 x 4 x 20 = 80 f16", "8 x 16 x 20 = 2560 f16", "32x"],
    ["V cache", "1 x 20 x 4 = 80 f16", "8 x 20 x 16 = 2560 f16", "32x"],
    ["Score buffer", "4 x 1 x 20 = 80 f16", "1 x 20 = 20 f16", "0.25x"],
    ["O / UP / GATE / DOWN", "identical", "identical", "1x"],
]
add_table(sl, 0.5, 1.5, 12.3, rows, [2.8, 3.8, 3.8, 1.2])

add_card(sl, 0.5, 5.3, 12.3, 1.8, RGBColor(0x33, 0x1C, 0x1C))
add_textbox(sl, 0.7, 5.4, 11.8, 0.4, "Key takeaway", font_size=18, color=RED, bold=True)
add_bullet_list(sl, 0.7, 5.8, 11.8, 1.2, [
    "WSE-3-GQA uses 32x less memory for K/V weights and cache (for 4:1 GQA ratio)",
    "The savings scale directly with the GQA ratio (n_heads / n_kv_heads)",
    "Critical for scaling to large models on fixed-SRAM PEs \u2014 every byte counts",
], font_size=14, color=LIGHT)
slide_number(sl, 6, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 7: QKV Projection & Tile Layout
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "QKV Projection & Tile Layout")

# Left: WSE-3-GQA
add_card(sl, 0.5, 1.4, 5.9, 5.5, RGBColor(0x1C, 0x33, 0x2C))
add_textbox(sl, 0.7, 1.5, 5.5, 0.35, "WSE-3-GQA: Asymmetric QKV", font_size=18, color=ACCENT_1, bold=True)
add_code_block(sl, 0.7, 2.0, 5.5, 1.8,
    "QKV_tile: [bsz * (dim_p_pe + 2*kv_dim_p_pe)]\n\n"
    "Q: offset=0,      extent=bsz*dim_p_pe     [16]\n"
    "K: offset=bsz*16,  extent=bsz*kv_dim_p_pe  [4]\n"
    "V: offset=bsz*20,  extent=bsz*kv_dim_p_pe  [4]\n\n"
    "K proj: Kt=dim_p_pe, Nt=kv_dim_p_pe  (16->4)",
    font_size=12)
add_bullet_list(sl, 0.7, 4.0, 5.5, 2.5, [
    "K/V projections output smaller kv_dim_p_pe",
    "QKV reduce: dim + 2*kv_dim elements",
    "Less communication volume",
    "Asymmetry adds complexity",
], font_size=13, color=LIGHT)

# Right: WSE-3-GQA-cheng
add_card(sl, 6.9, 1.4, 5.9, 5.5, RGBColor(0x2C, 0x1C, 0x33))
add_textbox(sl, 7.1, 1.5, 5.5, 0.35, "WSE-3-GQA-cheng: Symmetric QKV", font_size=18, color=ACCENT_3, bold=True)
add_code_block(sl, 7.1, 2.0, 5.5, 1.8,
    "QKV_tile: [bsz * 3 * dim_p_pe]\n\n"
    "Q: offset=0,       extent=bsz*dim_p_pe  [16]\n"
    "K: offset=bsz*16,  extent=bsz*dim_p_pe  [16]\n"
    "V: offset=bsz*32,  extent=bsz*dim_p_pe  [16]\n\n"
    "K proj: Kt=dim_p_pe, Nt=dim_p_pe  (16->16)",
    font_size=12)
add_bullet_list(sl, 7.1, 4.0, 5.5, 2.5, [
    "K/V projections output full dim_p_pe",
    "QKV reduce: 3 * dim elements",
    "More communication volume",
    "Simpler, uniform layout",
], font_size=13, color=LIGHT)
slide_number(sl, 7, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 8: Attention Score Computation
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "Attention Score Computation", "Where the two GQA strategies diverge most visibly")

# Left
add_card(sl, 0.5, 1.4, 5.9, 5.5, RGBColor(0x1C, 0x33, 0x2C))
add_textbox(sl, 0.7, 1.5, 5.5, 0.35, "WSE-3-GQA: Multi-Group", font_size=18, color=ACCENT_1, bold=True)
add_bullet_list(sl, 0.7, 1.95, 5.5, 2.8, [
    "Iterates gqa_group_size Q groups per batch",
    "All groups share same K cache",
    "Variable-length: Nt = iter_num (grows)",
    "Strided K access (max_seq_len_p_pe stride)",
    "KV-head-scoped reduction (axis=3)",
    "Score: [gqa_group_size x bsz x iter_num]",
], font_size=13, color=LIGHT)
add_code_block(sl, 0.7, 4.3, 5.5, 2.3,
    "for b in bsz:\n"
    "  for g in gqa_group_size:\n"
    "    score[g,b,:] = Q[b,g,:] @ K[b,:,:]^T\n\n"
    "all_reduce_kv_head_scoped(score)\n"
    "// New axis mode + new reduce fn",
    font_size=12)

# Right
add_card(sl, 6.9, 1.4, 5.9, 5.5, RGBColor(0x2C, 0x1C, 0x33))
add_textbox(sl, 7.1, 1.5, 5.5, 0.35, "WSE-3-GQA-cheng: Single GEMV", font_size=18, color=ACCENT_3, bold=True)
add_bullet_list(sl, 7.1, 1.95, 5.5, 2.8, [
    "Single Q @ K^T per PE (no multi-group)",
    "K cache selected via DSD offset",
    "Fixed-length: Nt = seq_len_p_pe (always full)",
    "Standard GEMV with DSD snapshot",
    "Standard global reduction (axis=0)",
    "Score: [bsz x seq_len_p_pe]",
], font_size=13, color=LIGHT)
add_code_block(sl, 7.1, 4.3, 5.5, 2.3,
    "xk_offset = kv_head_id * dim * seq_len\n"
    "right_matrix_dsd = offset(ptr_XKCache,\n"
    "                          xk_offset)\n\n"
    "score = Q @ K_cache[kv_head_id]^T\n"
    "all_reduce_bsz_seq_len(score)  // standard",
    font_size=12)
slide_number(sl, 8, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 9: KV Cache Management
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "KV Cache Management")

# Left
add_card(sl, 0.5, 1.4, 5.9, 5.5, RGBColor(0x1C, 0x33, 0x2C))
add_textbox(sl, 0.7, 1.5, 5.5, 0.35, "WSE-3-GQA: Full Autoregressive", font_size=18, color=ACCENT_1, bold=True)
add_bullet_list(sl, 0.7, 2.0, 5.5, 1.8, [
    "process_kv(): writes K,V to cache each step",
    "Distributed: only py == step % P writes",
    "iter_num tracks cached KV pairs (grows 1..max)",
    "Prefill support: iter_num starts at prefill_len",
    "Cache uses reduced kv_dim_p_pe",
], font_size=13, color=LIGHT)
add_code_block(sl, 0.7, 3.9, 5.5, 2.7,
    "var step: i16 = 0;\n"
    "var iter_num: i16 = prefill_len_p_pe;\n\n"
    "fn process_kv() void {\n"
    "  if (py == step % P) {\n"
    "    copy K -> XKCache[iter_num]\n"
    "    copy V -> XVCache[iter_num]\n"
    "    iter_num += 1;\n"
    "  }\n"
    "}",
    font_size=12)

# Right
add_card(sl, 6.9, 1.4, 5.9, 5.5, RGBColor(0x2C, 0x1C, 0x33))
add_textbox(sl, 7.1, 1.5, 5.5, 0.35, "WSE-3-GQA-cheng: None", font_size=18, color=ACCENT_3, bold=True)
add_bullet_list(sl, 7.1, 2.0, 5.5, 2.5, [
    "No process_kv() function",
    "No step / iter_num state variables",
    "No prefill_len_p_pe parameter",
    "KV cache assumed pre-filled by host",
    "Attention always uses full seq_len_p_pe",
    "",
    "Suitable for benchmarking or single-step eval",
    "Cannot do multi-step autoregressive decoding",
], font_size=13, color=LIGHT)
slide_number(sl, 9, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 10: Communication Library
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "Communication Library Extensions", "WSE-3-GQA adds 57% more code to comm_pe.csl (1708 vs 1086 lines)")

rows = [
    ["Reduction Function", "GQA", "cheng", "Purpose"],
    ["all_reduce_bsz", "\u2713", "\u2713", "RMSNorm scalar"],
    ["all_reduceMax_bsz", "\u2713", "\u2713", "Softmax max"],
    ["all_reduce_bsz_g", "\u2713", "\u2717", "Group-extended scalar (gqa_group_size*bsz)"],
    ["all_reduceMax_bsz_g", "\u2713", "\u2717", "Group-extended max"],
    ["all_reduce_bsz_dim", "\u2713", "\u2713", "GEMV output"],
    ["all_reduce_bsz_dim_QKV_fusion", "\u2713", "\u2713", "Fused Q,K,V (asymmetric in GQA)"],
    ["all_reduce_bsz_g_var_seq_len_kv_head_scoped", "\u2713", "\u2717", "Var-len group scores, KV-head scoped"],
    ["all_reduce_bsz_var_seq_len_strided", "\u2713", "\u2717", "Variable-length strided scores"],
    ["all_reduce_bsz_ffn_dim_ZZ_fusion", "\u2713", "\u2713", "Fused UP+GATE"],
    ["reconfig_allreduce_axis(3)", "\u2713", "\u2717", "KV-head-scoped X-axis mode"],
]
add_table(sl, 0.5, 1.4, 12.3, rows, [4.5, 0.7, 0.7, 5.0])

add_card(sl, 0.5, 6.0, 12.3, 1.1)
add_textbox(sl, 0.7, 6.1, 11.8, 0.9,
    "axis=3 configures routing so reduction occurs only within pes_p_kv_head PEs belonging to the same KV head group. "
    "This requires additional routing topology in comm_layout.csl (575 vs 406 lines).",
    font_size=13, color=GRAY)
slide_number(sl, 10, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 11: GEMV Primitives
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "GEMV Primitives")

rows = [
    ["Capability", "WSE-3-GQA", "WSE-3-GQA-cheng"],
    ["Fixed-size GEMV", "vecmat_computation()", "vecmat_computation()"],
    ["Variable-len (growing cache)", "vecmat_computation_var_right()\nvecmat_computation_var_right_2d_strided()", "Not supported"],
    ["Runtime offset preservation", "Not needed (no replicated heads)", "vecmat_computation_dsd() (DSD snapshot)"],
]
add_table(sl, 0.5, 1.5, 12.3, rows, [3.0, 4.65, 4.65])

# Code comparison
add_textbox(sl, 0.7, 3.8, 5.5, 0.35, "Variable-len strided (Q @ K^T)", font_size=14, color=ACCENT_1, bold=True)
add_code_block(sl, 0.5, 4.2, 5.9, 2.8,
    "// WSE-3-GQA: 1-element-at-a-time\n"
    "// strided access through K cache\n"
    "fn vecmat_var_right_2d_strided(\n"
    "    right_stride: i16) void {\n"
    "  left_dsd = set_length(1); // 1 element\n"
    "  for (b in bsz):\n"
    "    for (k in Kt): // kv_dim_p_pe\n"
    "      @map(gemv_step, left_dsd);\n"
    "      right += right_stride; // skip\n"
    "}",
    font_size=11)

add_textbox(sl, 7.1, 3.8, 5.5, 0.35, "DSD snapshot (offset preservation)", font_size=14, color=ACCENT_3, bold=True)
add_code_block(sl, 6.9, 4.2, 5.9, 2.8,
    "// WSE-3-GQA-cheng: snapshot + restore\n"
    "fn vecmat_computation_dsd() void {\n"
    "  // Preserve runtime GQA offset\n"
    "  right_matrix_dsd_snap = right_matrix_dsd;\n\n"
    "  for (b in bsz):\n"
    "    // Restore offset each batch row\n"
    "    right_matrix_dsd = snap;\n"
    "    @map(gemv_step, left_dsd);\n"
    "}",
    font_size=11)
slide_number(sl, 11, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 12: Bug Comparison
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "Bug Comparison & Fixes")

# SILU bug
add_card(sl, 0.5, 1.4, 12.3, 2.8, RGBColor(0x33, 0x1C, 0x1C))
add_textbox(sl, 0.7, 1.5, 11.8, 0.35, "Bug: SILU iteration (both WSE-3-GQA and GQA-cheng's base variant)", font_size=16, color=RED, bold=True)

add_code_block(sl, 0.7, 1.95, 5.3, 1.5,
    "// WSE-3-GQA (BUGGY)\n"
    "for (i in bsz*dim_p_pe):  // WRONG count\n"
    "  ZZ_tile[i] = silu(ZZ_tile[i])\n"
    "// Applies to z1 region, NOT z2",
    font_size=12)

add_code_block(sl, 6.5, 1.95, 6.0, 1.5,
    "// WSE-3-GQA-cheng full (FIXED)\n"
    "offset = bsz * ffn_dim_p_pe  // z2 start\n"
    "for (i in bsz*ffn_dim_p_pe):  // RIGHT count\n"
    "  ZZ_tile[offset+i] = silu(ZZ_tile[offset+i])",
    font_size=12)

add_textbox(sl, 0.7, 3.6, 11.8, 0.4,
    "Harmless when dim_p_pe == ffn_dim_p_pe (test configs), but wrong for real models (Llama-8B: dim_p_pe=16, ffn_dim_p_pe=56)",
    font_size=12, color=GRAY)

# RoPE + DOWN
rows = [
    ["Issue", "WSE-3-GQA", "WSE-3-GQA-cheng (full)", "Impact"],
    ["SILU iteration", "BUG: bsz*dim_p_pe, index 0", "FIXED: bsz*ffn_dim_p_pe, offset", "Wrong activation for real configs"],
    ["DOWN weight shape", "[dim_p_pe * ffn_dim_p_pe]", "[ffn_dim_p_pe * dim_p_pe] (semantic fix)", "Same buffer size, clearer intent"],
    ["QKV input", "Uses X_norm (correct)", "Base: X (bug), Full: X_norm (fixed)", "Bypasses RMSNorm in base variant"],
    ["RoPE", "Commented out (strided-DSD bug)", "Enabled in full variant", "Missing position encoding"],
]
add_table(sl, 0.5, 4.3, 12.3, rows, [2.0, 3.3, 3.6, 3.4])
slide_number(sl, 12, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 13: Pipeline Side-by-Side
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "Decode Pipeline Side-by-Side")

rows = [
    ["Step", "WSE-3-GQA", "WSE-3-GQA-cheng (full)"],
    ["1-2", "rmsnorm_x(), xq_matvec [Nt=dim]", "rmsnorm_x(), xq_matvec [Nt=dim]"],
    ["3-4", "xk/xv_matvec [Nt=kv_dim_p_pe]", "xk/xv_matvec [Nt=dim, DSD offset]"],
    ["5", "all_reduce QKV [dim + 2*kv_dim]", "all_reduce QKV [3*dim]"],
    ["6-8", "reconfig(0), RoPE commented out", "reconfig(0), xq_rope(), xk_rope()"],
    ["9", "process_kv() [var-len cache write]", "\u2014 (no KV cache management)"],
    ["10", "reconfig(3) [KV-head-scoped]", "\u2014 (no axis 3)"],
    ["11", "score_matvec [multi-group, var-len]", "score_matvec [single, fixed-len, offset]"],
    ["12-13", "reconfig(1), softmax_all_groups", "reconfig(1), softmax [standard]"],
    ["14", "output_matvec_all_groups [multi-grp]", "output_matvec [single, offset]"],
    ["15-17", "reconfig(0), o_matvec, residual", "reconfig(0), o_matvec, residual"],
    ["18-22", "reconfig(1), rmsnorm_z, UP+GATE fused", "reconfig(1), rmsnorm_z, UP+GATE fused"],
    ["23-28", "z2_silu [BUG], z3, DOWN, residual", "z2_silu [FIXED], z3, DOWN, residual"],
    ["29", "step += 1", "\u2014 (no step)"],
]
add_table(sl, 0.3, 1.3, 12.7, rows, [1.0, 5.5, 5.5])
slide_number(sl, 13, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 14: Code Complexity
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "Code Complexity & Production Readiness")

rows = [
    ["Metric", "WSE-3-GQA", "WSE-3-GQA-cheng (full)"],
    ["decode.csl", "1282 lines", "942 lines"],
    ["comm_pe.csl", "1708 lines", "1086 lines"],
    ["comm_layout.csl", "575 lines", "406 lines"],
    ["launch_sim.py", "~2000 lines", "~100 lines"],
    ["Total CSL source", "~3565 lines", "~2434 lines"],
    ["GEMV variants", "3", "2"],
    ["Reduction functions", "~12", "~8"],
    ["Axis modes", "3 (0, 1, 3)", "2 (0, 1)"],
    ["Model configs", "Llama-8B (4:1 GQA), Qwen-72B", "1 test config (2:1 GQA)"],
    ["Reference validation", "Full Python reference impl", "None"],
    ["Debug exports", "7 intermediate snapshots", "1 (debug only)"],
]
add_table(sl, 0.5, 1.4, 12.3, rows, [2.8, 4.75, 4.75])

add_card(sl, 0.5, 6.3, 12.3, 0.8)
add_textbox(sl, 0.7, 6.35, 11.8, 0.6,
    "WSE-3-GQA is ~46% larger in total CSL code. Growth is driven by KV-head-scoped reduction machinery and multi-group attention loops.",
    font_size=14, color=GRAY)
slide_number(sl, 14, TOTAL)

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 15: Key Takeaways
# ══════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
title_bar(sl, "Key Takeaways")

items = [
    ("Fundamentally different strategies",
     "Dimensional GQA (compile-time dims) vs Replicated-Heads GQA (runtime offset). Neither is strictly superior."),
    ("Memory efficiency: GQA wins big",
     "32x less K/V memory for Llama-8B 4:1 GQA. Critical for scaling on fixed-SRAM PEs."),
    ("Code complexity: cheng is simpler",
     "~46% less CSL code. GQA achievable with minimal changes (DSD offset + snapshot) vs new comm infrastructure."),
    ("Feature completeness: GQA is ahead",
     "KV cache management, variable-length attention, prefill support, debug instrumentation. Cheng lacks all of these."),
    ("Shared bugs need backporting",
     "SILU fix (iteration count + offset) from GQA-cheng should be backported to WSE-3-GQA."),
    ("Comm library is the biggest delta",
     "KV-head-scoped reductions (axis=3) and group-extended reduce functions represent major engineering in WSE-3-GQA."),
    ("Potential hybrid approach",
     "Use GQA's memory-efficient dimensional strategy + cheng's simpler DSD technique for edge cases with rounding issues."),
]

y = 1.4
for i, (title, desc) in enumerate(items):
    c = ACCENT_1 if i % 2 == 0 else ACCENT_2
    add_textbox(sl, 0.7, y, 0.3, 0.3, f"{i+1}.", font_size=16, color=c, bold=True)
    tf = add_textbox(sl, 1.1, y, 11.5, 0.25, title, font_size=15, color=c, bold=True)
    add_textbox(sl, 1.1, y + 0.3, 11.5, 0.35, desc, font_size=13, color=GRAY)
    y += 0.78

slide_number(sl, 15, TOTAL)

# ── Save ───────────────────────────────────────────────────────────────────
out_path = "/home/lexu/WaferLLM/Decode/GQA_Implementation_Comparison.pptx"
prs.save(out_path)
print(f"Saved to {out_path}")
