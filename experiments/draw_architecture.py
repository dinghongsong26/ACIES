"""Render the ContrAuto architecture figure (figure/ContrAuto.png) — publication style + emoji.

Two swimlanes (task / physical) as soft tinted bands; white cards with category-colored borders,
drop shadows, and a small COLOR EMOJI badge per card (rendered via Apple Color Emoji through
Pillow, since matplotlib's own fonts have no color emoji); one clean synthesis feedback edge;
separated repair edges; a real simulation thumbnail as the result.

Run:  python experiments/draw_architecture.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INK = "#23303A"
LLM, SYM, PHY, OUT, EXEC, GATE = "#3B6FB0", "#7E6BB5", "#2E9E8F", "#4E9A5A", "#D08A3E", "#C0504D"
BAND_TASK, BAND_PHYS = "#EEF3FB", "#EDF6F0"
FB, REPAIR = "#7C8A97", "#C0504D"

# ── color-emoji rendering via Apple Color Emoji (Pillow) ────────────────────────
_EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
_emoji_cache = {}


def _emoji_arr(ch):
    if ch in _emoji_cache:
        return _emoji_cache[ch]
    f = ImageFont.truetype(_EMOJI_FONT, 160)
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((12, 8), ch, font=f, embedded_color=True)
    bb = im.getbbox()
    arr = np.asarray(im.crop(bb)) if bb else np.asarray(im)
    _emoji_cache[ch] = arr
    return arr


fig = plt.figure(figsize=(17.8, 6.1), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 178); ax.set_ylim(0, 61); ax.set_aspect("equal"); ax.axis("off")
XLIM = 178
SHADOW = [pe.withSimplePatchShadow(offset=(1.1, -1.1), shadow_rgbFace="#9aa3ad", alpha=0.30)]


def badge(ch, x, y, zoom=0.115):
    ax.add_artist(AnnotationBbox(OffsetImage(_emoji_arr(ch), zoom=zoom), (x, y),
                                 frameon=False, zorder=6, box_alignment=(0.5, 0.5)))


def band(x, y, w, h, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2.2",
                                fc=fill, ec="none", zorder=1))


def card(cx, cy, w, h, border, title, sub="", emoji="", fs=7.6):
    r = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.2,rounding_size=1.1",
                       fc="white", ec=border, lw=1.7, zorder=3)
    r.set_path_effects(SHADOW)
    ax.add_patch(r)
    if sub:
        ax.text(cx, cy + 0.9, title, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold", zorder=4)
        ax.text(cx, cy - 1.7, sub, ha="center", va="center", fontsize=fs - 1.4, color="#5b6b78",
                zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold", zorder=4)
    if emoji:
        badge(emoji, cx - w / 2 + 0.4, cy + h / 2 - 0.2)        # top-left corner badge


def arrow(p, q, label="", color="#46545f", lw=1.4, rad=0.0, fs=6.3, ls="-", loff=(0, 1.6)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, ls=ls,
                                 zorder=2, shrinkA=0, shrinkB=0))
    if label:
        mx, my = (p[0] + q[0]) / 2 + loff[0], (p[1] + q[1]) / 2 + loff[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs, color=color,
                style="italic", zorder=4)


# ── title ───────────────────────────────────────────────────────────────────────
ax.text(3, 58.4, "ContrAuto", fontsize=15, fontweight="bold", color=INK)
ax.text(3, 55.2, "automated two-layer, contract-based design for multi-robot cyber-physical systems",
        fontsize=8.2, color="#5b6b78")

# ── swimlanes ─────────────────────────────────────────────────────────────────────
band(30, 29.5, 99, 21, BAND_TASK)
band(30, 7.5, 99, 17, BAND_PHYS)
badge("🧠", 33.2, 47.8, zoom=0.085); ax.text(35.4, 47.6, "TASK LAYER", fontsize=8.6, fontweight="bold", color=LLM)
ax.text(49.6, 47.6, "·  discrete — LLM agents + CHASE", fontsize=7.4, color="#5b6b78")
badge("🦾", 33.2, 22.4, zoom=0.085); ax.text(35.4, 22.2, "PHYSICAL LAYER", fontsize=8.6, fontweight="bold", color=PHY)
ax.text(54.5, 22.2, "·  continuous — CBF contracts  (CBC = Control Barrier · RC = Reachability)",
        fontsize=7.4, color="#5b6b78")

# ── THREE separate input cards, each routed to the layer(s) it feeds ────────────────
ax.text(14, 51.4, "INPUTS  (one task YAML)", ha="center", fontsize=7.6, fontweight="bold", color="#5b6b78")
card(14, 44, 24, 8, "#8696A6", "Natural language", "requirements + context", emoji="📝")
card(14, 31, 24, 8, "#8696A6", "Architecture", "nodes + comm. edges", emoji="🔗")
card(14, 18, 24, 8, "#8696A6", "Map  (+ physics)", "layout / bounds · dynamics", emoji="🗺️")
# routing — colored by destination lane:  blue = TASK layer, teal = PHYSICAL layer
arrow((26.3, 44), (37.0, 43.0), "to task", color=LLM, rad=0.05, fs=6.0)
arrow((26.3, 32.5), (37.0, 41.5), "to task", color=LLM, rad=0.10, fs=6.0, loff=(0.5, 1.8))     # architecture → task
arrow((26.3, 29.5), (37.0, 16.8), "to physical", color=PHY, rad=-0.12, fs=6.0, loff=(-1.2, 0.2))  # architecture → physical
arrow((26.3, 18), (37.0, 15.4), "to physical", color=PHY, rad=-0.05, fs=6.0, loff=(0, -1.7))

# ── TASK LAYER (snake: row1 L→R, row2 R→L) ─────────────────────────────────────────
card(46, 42.5, 16, 7, LLM, "Translator", "LLM agent", emoji="🌐")
card(67, 42.5, 12, 6, SYM, "LTL check", emoji="✅")
card(88, 42.5, 16, 7, LLM, "Decomposer", "LLM agent", emoji="🧩")
arrow((54.2, 42.5), (60.8, 42.5), "LTL")
arrow((73.2, 42.5), (79.8, 42.5), "pass")

card(88, 33, 23, 7, SYM, "CHASE checks", "consistency·compat·refine", emoji="🔍")
card(63, 33, 13, 6, SYM, "SLUGS", "synthesis", emoji="⚙️")
card(45, 33, 12, 6.5, GATE, "well-sep", "gate", emoji="🚦")
arrow((88, 39.0), (88, 36.6), "contract", loff=(5.2, 0))
arrow((76.3, 33), (69.7, 33), "pass")
arrow((56.4, 33), (51.2, 33), "FSM")

# handoff to the physical layer
arrow((45, 29.6), (44, 19.6), "well-sep ✓", loff=(5.2, 0), fs=6.0, color=OUT, lw=1.6)

# ── PHYSICAL LAYER (row L→R) ───────────────────────────────────────────────────────
card(44, 16, 11, 6.5, OUT, "FSM", "controller", emoji="🤖")
card(64, 16, 14, 7, PHY, "Planner", "A* agent", emoji="🧭")
card(85, 16, 15, 7, PHY, "CBF-QP", "CBC ∧ RC", emoji="🛡️")
card(106, 16, 13, 7, OUT, "certificate", "≤ deadline", emoji="📜")
card(122, 16, 11, 6.5, EXEC, "execution", emoji="🏁")
arrow((49.6, 16), (56.8, 16), "target")
arrow((71.2, 16), (77.3, 16), "route→RC")
arrow((92.7, 16), (99.3, 16), "reach/avoid")
arrow((112.7, 16), (116.3, 16), "")
arrow((127.6, 16), (130.3, 16), "")

# ── synthesis feedback (RED, dashed): EACH failed check / synthesis / gate → Decomposer ─
ax.text(64, 50.8, "on any failure → Decomposer regenerates  (episodic memory accumulates)",
        ha="center", fontsize=6.4, style="italic", color=REPAIR)
# CHASE checks → Decomposer (short, right; parallel to the forward "contract" arrow)
ax.add_patch(FancyArrowPatch((92, 36.7), (92, 38.9), arrowstyle="-|>", mutation_scale=9,
             connectionstyle="arc3,rad=0.0", color=REPAIR, lw=1.3, ls=(0, (4, 2)), zorder=5))
ax.text(95.4, 37.8, "refine\nfail", fontsize=5.6, style="italic", color=REPAIR, va="center")
# SLUGS synthesis → Decomposer  (UNREALIZABLE)
ax.add_patch(FancyArrowPatch((64, 36.1), (82, 39.4), arrowstyle="-|>", mutation_scale=9,
             connectionstyle="arc3,rad=-0.28", color=REPAIR, lw=1.3, ls=(0, (4, 2)), zorder=5))
ax.text(71.5, 40.6, "unrealizable", fontsize=5.8, style="italic", color=REPAIR)
# well-sep gate → Decomposer  (DEGENERATE) — high arc above the boxes
ax.add_patch(FancyArrowPatch((45, 36.4), (80.3, 46.2), arrowstyle="-|>", mutation_scale=10,
             connectionstyle="arc3,rad=-0.52", color=REPAIR, lw=1.3, ls=(0, (4, 2)), zorder=5))
ax.text(55, 49.3, "degenerate", fontsize=5.8, style="italic", color=REPAIR)

# ── repair edges (physical below the row; task on the right) ───────────────────────
arrow((122, 12.7), (64, 12.7), "reassign · relax · yield   (physical repair)", color=REPAIR,
      rad=0.32, lw=1.4, fs=6.3, loff=(0, -2.6))
ax.add_patch(FancyArrowPatch((127.2, 16.5), (95.5, 39.2), arrowstyle="-|>", mutation_scale=10,
             connectionstyle="arc3,rad=0.52", color=REPAIR, lw=1.4, ls=(0, (5, 2)), zorder=2))
ax.text(125, 28, "structural →\nre-synthesize\n(task repair)", ha="center", fontsize=6.3,
        style="italic", color=REPAIR)

# ── result panel (the certified outcome — bigger, landscape) ────────────────────────
sx0, sx1, sy0, sy1 = 131, 176, 10, 40           # landscape, matches the frame's ~1.5:1
fr = FancyBboxPatch((sx0, sy0), sx1 - sx0, sy1 - sy0, boxstyle="round,pad=0.3,rounding_size=1.6",
                    fc="#161922", ec="#3a3f4d", lw=1.4, zorder=3); fr.set_path_effects(SHADOW)
ax.add_patch(fr)
badge("✅", sx0 + 6.2, sy1 + 2.4, zoom=0.085)
ax.text((sx0 + sx1) / 2 + 1.2, sy1 + 2.2, "Result — requirements met", ha="center", fontsize=8.6,
        fontweight="bold", color=INK, zorder=5)
ax.text((sx0 + sx1) / 2, sy0 - 2.2, "all delivered · within deadline · collision- & mutex-safe",
        ha="center", fontsize=7.0, color="#5b6b78", style="italic", zorder=5)
sim_path = os.path.join(ROOT, "figure", "_sim_panel.png")
if os.path.exists(sim_path):
    simax = fig.add_axes([(sx0 + 1) / XLIM, (sy0 + 1) / 61, (sx1 - sx0 - 2) / XLIM,
                          (sy1 - sy0 - 2) / 61])
    simax.imshow(mpimg.imread(sim_path)); simax.axis("off")

out = os.path.join(ROOT, "figure", "ContrAuto.png")
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
print("wrote", out)
