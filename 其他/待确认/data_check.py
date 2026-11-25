# draw_pipeline.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams

# -----------------------------
# Style (CVPR-ish: clean, flat, vector-friendly)
# -----------------------------
rcParams["font.family"] = "DejaVu Sans"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

def add_box(ax, x, y, w, h, title, subtitle=None, fc="#F7F7F7", ec="#222222",
            lw=1.2, radius=0.08, title_size=12, sub_size=10, align="center"):
    """Add rounded box with optional subtitle. (x,y) is lower-left in figure coords."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(patch)

    # Title
    tx = x + (w/2 if align == "center" else 0.03*w)
    ty = y + h*0.62
    ax.text(tx, ty, title, ha=align, va="center", fontsize=title_size, weight="bold", color="#111111")

    # Subtitle
    if subtitle:
        ax.text(tx, y + h*0.28, subtitle, ha=align, va="center", fontsize=sub_size, color="#333333")
    return patch

def add_arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.4, head=10, rad=0.0):
    """Arrow from (x1,y1) to (x2,y2)."""
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=f"-|>,head_width={head*0.01},head_length={head*0.015}",
        mutation_scale=18,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(arr)
    return arr

def mid_top(patch):
    x, y = patch.get_x(), patch.get_y()
    w, h = patch.get_width(), patch.get_height()
    return (x + w/2, y + h)

def mid_bottom(patch):
    x, y = patch.get_x(), patch.get_y()
    w = patch.get_width()
    return (x + w/2, y)

def mid_left(patch):
    x, y = patch.get_x(), patch.get_y()
    h = patch.get_height()
    return (x, y + h/2)

def mid_right(patch):
    x, y = patch.get_x(), patch.get_y()
    w, h = patch.get_width(), patch.get_height()
    return (x + w, y + h/2)

# -----------------------------
# Figure canvas
# -----------------------------
fig = plt.figure(figsize=(16, 9))
ax = plt.axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# -----------------------------
# Colors
# -----------------------------
C_GROUP = "#FFF3D6"      # light warm
C_STAGE = "#E8F1FF"      # light blue
C_HLOC  = "#DDEBFF"
C_REF   = "#EFEFEF"
C_EVAL  = "#E9F7E6"
C_ACC   = "#CFE3FF"
C_COMP  = "#D6F0D0"

# -----------------------------
# Top: 8 groups layout (2 rows x 4 cols)
# -----------------------------
group_w, group_h = 0.20, 0.12
x0 = 0.10
gap_x = 0.03
y_row1 = 0.84  # 1 FPS row
y_row2 = 0.69  # 3 FPS row

# Exact ordering you requested:
# Row1: P0 1FPS, P1 1FPS, P2 1FPS, P3 1FPS
# Row2: P0 3FPS, P1 3FPS, P2 3FPS, P3 3FPS
labels_row1 = [("Config P0", "1 FPS"), ("Config P1", "1 FPS"), ("Config P2", "1 FPS"), ("Config P3", "1 FPS")]
labels_row2 = [("Config P0", "3 FPS"), ("Config P1", "3 FPS"), ("Config P2", "3 FPS"), ("Config P3", "3 FPS")]

group_boxes = []
for i, (t, s) in enumerate(labels_row1):
    bx = x0 + i*(group_w + gap_x)
    group_boxes.append(add_box(ax, bx, y_row1, group_w, group_h, t, s, fc=C_GROUP))

for i, (t, s) in enumerate(labels_row2):
    bx = x0 + i*(group_w + gap_x)
    group_boxes.append(add_box(ax, bx, y_row2, group_w, group_h, t, s, fc=C_GROUP))

# -----------------------------
# Middle: Query Generation
# -----------------------------
query_box = add_box(
    ax, 0.32, 0.56, 0.36, 0.075,
    "Query Generation (Equirect → Rectilinear Views)",
    "FFmpeg v360: output=rectilinear, yaw/pitch grid, h_fov=v_fov, size",
    fc=C_STAGE, title_size=13, sub_size=10
)

# Connect group boxes -> query
for gb in group_boxes:
    x1, y1 = mid_bottom(gb)
    x2, y2 = mid_top(query_box)
    add_arrow(ax, x1, y1, x2, y2, lw=1.0, color="#666666")

# -----------------------------
# Left: Reference (UAV database / SfM)
# -----------------------------
ref_box = add_box(
    ax, 0.03, 0.18, 0.22, 0.32,
    "UAV Reference (COLMAP SfM)",
    "dbprefix model + db/xxx image names\nsymlink to flat uav_img_root",
    fc=C_REF, title_size=13, sub_size=10, align="center"
)

# -----------------------------
# HLOC Pipeline blocks
# -----------------------------
# Place blocks horizontally
bx0, by0 = 0.27, 0.36
bw, bh = 0.17, 0.09
gap = 0.02

b_global = add_box(ax, bx0 + 0*(bw+gap), by0, bw, bh,
                   "Global Descriptors",
                   "NetVLAD (db + query)",
                   fc=C_HLOC, title_size=12, sub_size=10)

b_pairs = add_box(ax, bx0 + 1*(bw+gap), by0, bw, bh,
                  "Top-K Retrieval Pairs",
                  "pairs_from_retrieval\n(query ↔ db)",
                  fc=C_HLOC, title_size=12, sub_size=10)

b_local = add_box(ax, bx0 + 2*(bw+gap), by0, bw, bh,
                  "Local Features",
                  "SuperPoint",
                  fc=C_HLOC, title_size=12, sub_size=10)

b_match = add_box(ax, bx0 + 3*(bw+gap), by0, bw, bh,
                  "Local Matching",
                  "SuperGlue",
                  fc=C_HLOC, title_size=12, sub_size=10)

b_pnp = add_box(ax, bx0 + 4*(bw+gap), by0, bw, bh,
                "Pose Estimation",
                "localize_sfm (PnP+RANSAC)",
                fc="#FFE8E8", title_size=12, sub_size=10)

# Connect query -> first HLOC block
add_arrow(ax, *mid_bottom(query_box), *mid_top(b_global), lw=1.3)

# Connect reference -> retrieval (to emphasize db side)
add_arrow(ax, *mid_right(ref_box), *mid_left(b_global), lw=1.3, rad=0.0)

# Chain arrows
add_arrow(ax, *mid_right(b_global), *mid_left(b_pairs))
add_arrow(ax, *mid_right(b_pairs), *mid_left(b_local))
add_arrow(ax, *mid_right(b_local), *mid_left(b_match))
add_arrow(ax, *mid_right(b_match), *mid_left(b_pnp))

# -----------------------------
# Right: Evaluation / Metrics
# -----------------------------
eval_box = add_box(ax, 0.78, 0.18, 0.19, 0.32,
                   "Evaluation (per group)",
                   "summary.jsonl + metrics.json",
                   fc=C_EVAL, title_size=13, sub_size=10)

acc_box = add_box(ax, 0.80, 0.34, 0.15, 0.13,
                  "Quality Metrics",
                  "num_inliers stats\nreproj error stats\ncenter spread (per pano frame)\nvalid_rate",
                  fc=C_ACC, title_size=12, sub_size=9)

comp_box = add_box(ax, 0.80, 0.20, 0.15, 0.11,
                   "Efficiency",
                   "runtime per stage\nwarnings / failures",
                   fc=C_COMP, title_size=12, sub_size=9)

add_arrow(ax, *mid_right(b_pnp), *mid_left(eval_box), lw=1.3)

# -----------------------------
# Title
# -----------------------------
ax.text(0.5, 0.965,
        "Ablation Pipeline for Ground-to-UAV Bridging via Panoramic Projection + HLOC Localization",
        ha="center", va="center", fontsize=16, weight="bold", color="#111111")

ax.text(0.5, 0.935,
        "8 groups = {P0,P1,P2,P3} × {1 FPS, 3 FPS}; compare pose reliability beyond localized_count",
        ha="center", va="center", fontsize=11, color="#333333")

# -----------------------------
# Export
# -----------------------------
plt.savefig("pipeline_cvpr_style.pdf", dpi=300, bbox_inches="tight")
plt.savefig("pipeline_cvpr_style.png", dpi=300, bbox_inches="tight")
print("Saved: pipeline_cvpr_style.pdf / pipeline_cvpr_style.png")
