"""Generate PowerPoint slides for WaferLLM Gap Analysis: Llama3-8B on WSE-3."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# -- Colors --
BG_DARK = RGBColor(0x1B, 0x1B, 0x2F)
BG_MED = RGBColor(0x23, 0x23, 0x3F)
ACCENT = RGBColor(0x4E, 0xC9, 0xB0)
ACCENT2 = RGBColor(0x56, 0x9C, 0xD6)
ACCENT3 = RGBColor(0xDC, 0xDC, 0xAA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
RED = RGBColor(0xF4, 0x47, 0x47)
YELLOW = RGBColor(0xFF, 0xD7, 0x00)
GREEN = RGBColor(0x4E, 0xC9, 0xB0)
ROW_EVEN = RGBColor(0x2A, 0x2A, 0x4A)
ROW_ODD = RGBColor(0x22, 0x22, 0x3C)
HEADER_BG = RGBColor(0x3A, 0x3A, 0x5C)


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_table(slide, left, top, width, rows_data, col_widths, header=True):
    """Add a styled table to a slide."""
    n_rows = len(rows_data)
    n_cols = len(rows_data[0])
    table_height = Inches(0.4) * n_rows

    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, table_height)
    table = shape.table

    # Set column widths
    for i, w in enumerate(col_widths):
        table.columns[i].width = w

    for row_idx, row_data in enumerate(rows_data):
        for col_idx, cell_text in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.text = str(cell_text)
            p.font.size = Pt(13)
            p.font.name = "Calibri"

            if row_idx == 0 and header:
                p.font.bold = True
                p.font.color.rgb = WHITE
                p.font.size = Pt(14)
                cell_fill = cell.fill
                cell_fill.solid()
                cell_fill.fore_color.rgb = HEADER_BG
            else:
                p.font.color.rgb = LIGHT_GRAY
                cell_fill = cell.fill
                cell_fill.solid()
                cell_fill.fore_color.rgb = ROW_EVEN if row_idx % 2 == 0 else ROW_ODD

            # Color-code priority or status columns
            text_lower = cell_text.lower() if isinstance(cell_text, str) else ""
            if "yes" in text_lower and col_idx > 0:
                p.font.color.rgb = GREEN
            elif "no" in text_lower and col_idx > 0:
                p.font.color.rgb = RED
            elif "disabled" in text_lower:
                p.font.color.rgb = YELLOW
            elif text_lower.startswith("p0"):
                p.font.color.rgb = RED
                p.font.bold = True
            elif text_lower.startswith("p1"):
                p.font.color.rgb = YELLOW
                p.font.bold = True
            elif text_lower.startswith("p2"):
                p.font.color.rgb = GREEN
                p.font.bold = True

            p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    return shape


# ============================================================
# SLIDE 1: Title
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
set_slide_bg(slide, BG_DARK)
add_text_box(slide, Inches(1), Inches(1.8), Inches(11), Inches(1.2),
             "WaferLLM Gap Analysis", font_size=44, color=ACCENT, bold=True,
             alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(1), Inches(3.0), Inches(11), Inches(0.8),
             "What's needed to run full Llama3-8B inference on WSE-3",
             font_size=24, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(1), Inches(4.2), Inches(11), Inches(0.6),
             "Prefill (WaferServe)  |  Decode (WSE-3-GQA)",
             font_size=18, color=ACCENT2, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(1), Inches(6.0), Inches(11), Inches(0.5),
             "April 2026", font_size=16, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 2: Current Implementation Status
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_text_box(slide, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
             "Current Implementation: Single Transformer Layer",
             font_size=32, color=ACCENT, bold=True)
add_text_box(slide, Inches(0.6), Inches(0.9), Inches(12), Inches(0.5),
             "Operations supported in Prefill/WSE-3 and Decode/WSE-3-GQA",
             font_size=16, color=LIGHT_GRAY)

rows = [
    ["Operation", "Prefill", "Decode (GQA)", "Notes"],
    ["RMSNorm (pre-attention)", "Yes", "Yes", ""],
    ["Q / K / V Projections", "Yes", "Yes", "GEMM (prefill) / GEMV (decode)"],
    ["RoPE", "Yes", "Disabled", "Strided DSD @fmulh bug in decode"],
    ["KV Cache Write", "No (allocated)", "Yes", "Incremental write in decode"],
    ["Causal Masking", "No", "N/A", "Autoregressive by nature in decode"],
    ["Score: Q @ K^T", "Yes", "Yes", "Variable-length in decode"],
    ["Softmax", "Yes", "Yes", "fast_exp() approx in decode"],
    ["Output: attn @ V", "Yes", "Yes", "4 variants in decode"],
    ["O-Projection", "Yes", "Yes", ""],
    ["Residual Add", "Yes", "Yes", ""],
    ["RMSNorm (pre-FFN)", "Yes", "Yes", ""],
    ["UP Projection (SwiGLU)", "Yes", "Yes", ""],
    ["GATE Projection (SwiGLU)", "Yes", "Yes", ""],
    ["SiLU Activation", "Yes", "Yes", ""],
    ["DOWN Projection", "Yes", "Yes", ""],
    ["FFN Residual Add", "Yes", "Yes", ""],
]

add_table(slide, Inches(0.4), Inches(1.4), Inches(12.5), rows,
          [Inches(3.0), Inches(1.5), Inches(1.8), Inches(6.2)])

# ============================================================
# SLIDE 3: What's Missing - Overview
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_text_box(slide, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
             "Missing Operators & Features for Llama3-8B",
             font_size=32, color=ACCENT, bold=True)
add_text_box(slide, Inches(0.6), Inches(0.9), Inches(12), Inches(0.5),
             "End-to-end inference requires these additional components beyond the single-layer implementation",
             font_size=16, color=LIGHT_GRAY)

rows = [
    ["Priority", "Operator / Feature", "Complexity", "Notes"],
    ["P0", "Fix RoPE in Decode", "Medium", "Debug strided DSD bug or restructure memory layout"],
    ["P0", "Causal Mask in Prefill", "Low", "Add -inf masking to upper triangle before softmax"],
    ["P0", "Multi-Layer Loop + Weight Mgmt", "High", "32 layers; weight streaming or full on-chip storage"],
    ["P0", "LM Head (Vocab Projection)", "High", "4096 -> 128,256 GEMV; largest single operation (~1 GB)"],
    ["P0", "Sampling (argmax minimum)", "Medium", "Distributed argmax reduction over 128K sharded logits"],
    ["P1", "Token Embedding Lookup", "Medium", "Host-side compute + transfer, or on-chip scatter"],
    ["P1", "Final RMSNorm", "Low", "Reuse existing RMSNorm kernel with separate weight vector"],
    ["P1", "Per-Layer KV Cache Management", "High", "32 x KV caches; memory allocation strategy needed"],
    ["P1", "Host <-> WSE Orchestration", "High", "Weight loading, token I/O loop, pipeline control"],
    ["P2", "Top-k / Top-p Sampling", "Medium", "More complex distributed reduction"],
    ["P2", "Tokenizer / Detokenizer", "Low", "Host-side; use existing libraries (tiktoken)"],
]

add_table(slide, Inches(0.4), Inches(1.4), Inches(12.5), rows,
          [Inches(0.9), Inches(3.5), Inches(1.4), Inches(6.7)])

# ============================================================
# SLIDE 4: Llama3-8B Full Pipeline Diagram
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_text_box(slide, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
             "Llama3-8B Inference Pipeline",
             font_size=32, color=ACCENT, bold=True)
add_text_box(slide, Inches(0.6), Inches(0.9), Inches(12), Inches(0.5),
             "Green = implemented  |  Red = missing  |  Yellow = partially implemented / broken",
             font_size=16, color=LIGHT_GRAY)

pipeline_steps = [
    ("Tokenizer\n(Host CPU)", RED, "BPE tokenization\n128K vocab"),
    ("Embedding\nLookup", RED, "128256 x 4096\n~1 GB table"),
    ("Transformer\nx 32 Layers", YELLOW, "Single layer done\nNeed loop + weights"),
    ("Final\nRMSNorm", RED, "Reuse kernel\nNew weights"),
    ("LM Head\n(Vocab Proj)", RED, "4096 -> 128256\nLargest GEMV"),
    ("Sampling\n(argmax/top-k)", RED, "Distributed\nreduction"),
    ("Detokenizer\n(Host CPU)", RED, "Token -> text"),
]

box_w = Inches(1.5)
box_h = Inches(1.2)
start_x = Inches(0.5)
y_top = Inches(2.0)
y_desc = Inches(3.5)
gap = Inches(0.3)

for i, (label, color, desc) in enumerate(pipeline_steps):
    x = start_x + i * (box_w + gap)

    # Box
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y_top, box_w, box_h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = BG_MED
    shape.line.color.rgb = color
    shape.line.width = Pt(2)
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = label
    p.font.size = Pt(13)
    p.font.color.rgb = color
    p.font.bold = True
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER
    tf.paragraphs[0].space_before = Pt(8)

    # Description below
    add_text_box(slide, x, y_desc, box_w, Inches(1.0), desc,
                 font_size=11, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)

    # Arrow between boxes
    if i < len(pipeline_steps) - 1:
        arrow_x = x + box_w
        arrow_y = y_top + box_h / 2
        connector = slide.shapes.add_shape(
            MSO_SHAPE.RIGHT_ARROW, arrow_x, arrow_y - Inches(0.1),
            gap, Inches(0.2))
        connector.fill.solid()
        connector.fill.fore_color.rgb = LIGHT_GRAY
        connector.line.fill.background()

# Transformer layer detail box
detail_y = Inches(4.8)
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.5), detail_y, Inches(12.3), Inches(2.2))
shape.fill.solid()
shape.fill.fore_color.rgb = BG_MED
shape.line.color.rgb = ACCENT2
shape.line.width = Pt(1)
tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Inside Each Transformer Layer (implemented):"
p.font.size = Pt(14)
p.font.color.rgb = ACCENT2
p.font.bold = True
p.font.name = "Calibri"

details = [
    "RMSNorm -> Q/K/V Proj -> RoPE* -> Score (Q@K^T) -> Causal Mask** -> Softmax -> Attn@V -> O-Proj -> Residual Add",
    "RMSNorm -> UP Proj -> GATE Proj -> SiLU -> DOWN Proj -> Residual Add",
    "",
    "* RoPE disabled in Decode (strided DSD bug)    ** Causal mask missing in Prefill"
]
for line in details:
    p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(12)
    p.font.color.rgb = YELLOW if line.startswith("*") else LIGHT_GRAY
    p.font.name = "Calibri"

# ============================================================
# SLIDE 5: P0 Details - Critical Path
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_text_box(slide, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
             "P0 Critical Items - Detail",
             font_size=32, color=RED, bold=True)

p0_items = [
    ("Fix RoPE in Decode", "Medium",
     "RoPE is disabled due to a strided DSD @fmulh bug. Without positional encoding, "
     "the model produces garbage. Options: (a) fix the strided DSD issue, "
     "(b) restructure memory layout to avoid strided access, (c) pre-apply RoPE on host."),
    ("Causal Mask in Prefill", "Low",
     "Current prefill computes full seq_len x seq_len attention with no mask. "
     "Need to set upper-triangle scores to -inf before softmax. "
     "Simple element-wise operation on the score matrix."),
    ("Multi-Layer Loop + Weight Management", "High",
     "Llama3-8B has 32 layers. Options: (a) weight streaming from host per layer (~14 GB total), "
     "(b) all weights on-chip (feasible on WSE-3 ~40 GB SRAM), "
     "(c) hybrid. Each layer needs separate weights and KV cache."),
    ("LM Head (Vocab Projection)", "High",
     "Final [1, 4096] x [4096, 128256] projection. The weight matrix alone is ~1 GB in f16. "
     "Llama3 ties embedding and LM head weights. Requires new sharding strategy for the "
     "extremely wide output dimension."),
    ("Sampling (argmax)", "Medium",
     "Distributed argmax over 128K logits sharded across PEs. "
     "Need a new reduction primitive that returns both max value and its index. "
     "Temperature scaling is a simple element-wise multiply before argmax."),
]

y = Inches(1.1)
for title, complexity, desc in p0_items:
    add_text_box(slide, Inches(0.6), y, Inches(10), Inches(0.4),
                 f"{title}  [{complexity}]", font_size=18, color=RED, bold=True)
    add_text_box(slide, Inches(0.8), y + Inches(0.35), Inches(11.5), Inches(0.8),
                 desc, font_size=13, color=LIGHT_GRAY)
    y += Inches(1.2)

# ============================================================
# SLIDE 6: P1/P2 Details
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_text_box(slide, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
             "P1 / P2 Items",
             font_size=32, color=YELLOW, bold=True)

p1_items = [
    ("P1", "Token Embedding Lookup", "Medium",
     "Convert token IDs to 4096-dim vectors. 128256 x 4096 table (~1 GB). "
     "Can be done on host and transferred, or implement on-chip indexed scatter."),
    ("P1", "Final RMSNorm", "Low",
     "Same kernel as existing RMSNorm, just needs to be invoked post-loop "
     "with a separate weight vector. Minimal new code."),
    ("P1", "Per-Layer KV Cache Management", "High",
     "32 layers x KV cache. At seq_len=8192: ~512 MB total. "
     "Need memory allocation strategy across PEs and lifetime management."),
    ("P1", "Host <-> WSE Orchestration", "High",
     "Full pipeline control: weight loading, token I/O per decode step, "
     "KV cache init from prefill. Extend launch scripts to loop."),
    ("P2", "Top-k / Top-p Sampling", "Medium",
     "More sophisticated sampling beyond argmax. Requires distributed "
     "partial sort or threshold-based filtering across PEs."),
    ("P2", "Tokenizer / Detokenizer", "Low",
     "Host-side only. Use tiktoken or sentencepiece. Not a WSE operator."),
]

y = Inches(1.1)
for prio, title, complexity, desc in p1_items:
    color = YELLOW if prio == "P1" else GREEN
    add_text_box(slide, Inches(0.6), y, Inches(10), Inches(0.4),
                 f"[{prio}] {title}  [{complexity}]", font_size=17, color=color, bold=True)
    add_text_box(slide, Inches(0.8), y + Inches(0.32), Inches(11.5), Inches(0.7),
                 desc, font_size=13, color=LIGHT_GRAY)
    y += Inches(1.0)

# ============================================================
# SLIDE 7: Memory Budget
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_text_box(slide, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
             "Llama3-8B Memory Budget on WSE-3",
             font_size=32, color=ACCENT, bold=True)

rows = [
    ["Component", "Size (f16)", "Per Layer", "Total (32 layers)"],
    ["Q / K / V / O Weights", "4 x 4096 x 4096 x 2B = 128 MB*", "128 MB", "~4 GB"],
    ["  (GQA: K,V are 4096x1024)", "Q,O: 64 MB + K,V: 16 MB = 96 MB", "96 MB", "~3 GB"],
    ["FFN Weights (UP+GATE+DOWN)", "3 x 4096 x 14336 x 2B = 336 MB", "336 MB", "~10.5 GB"],
    ["RMSNorm Weights", "2 x 4096 x 2B = 16 KB", "16 KB", "~0.5 MB"],
    ["KV Cache (seq=8192, bsz=1)", "2 x 8 x 128 x 8192 x 2B = 32 MB", "32 MB", "~1 GB"],
    ["Embedding / LM Head", "128256 x 4096 x 2B = 1 GB", "--", "1 GB (shared)"],
    ["Total", "", "", "~15.5 GB"],
]

add_table(slide, Inches(0.4), Inches(1.3), Inches(12.5), rows,
          [Inches(3.2), Inches(4.0), Inches(1.8), Inches(3.5)])

add_text_box(slide, Inches(0.6), Inches(5.5), Inches(12), Inches(1.5),
             "WSE-3 has ~40 GB distributed SRAM across all PEs.\n"
             "Llama3-8B at ~15.5 GB fits comfortably with room for activations and intermediates.\n"
             "All-on-chip strategy is feasible -- no weight streaming required.",
             font_size=16, color=ACCENT2)

# ============================================================
# Save
# ============================================================
out_path = "/mnt/raid0sata2/lexu/lexu/WaferLLM/Decode/WaferLLM_Gap_Analysis_Llama3_8B.pptx"
prs.save(out_path)
print(f"Saved to {out_path}")
