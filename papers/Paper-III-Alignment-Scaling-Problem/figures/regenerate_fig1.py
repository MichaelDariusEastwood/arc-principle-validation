#!/usr/bin/env python3
"""Regenerate fig1_equation.png with corrected title."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    'navy': '#1a365d',
    'dark': '#1a1a1a',
    'medium': '#4a4a4a',
    'light': '#6b7280',
    'bg_light': '#f8fafc',
    'border': '#e5e7eb',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

ax.text(5, 6.5, "The ARC Principle",
        ha='center', va='top', fontsize=14, fontweight='bold', color=COLORS['navy'])

ax.text(5, 5.2, r"$U = I \times R^{\alpha}$",
        ha='center', va='center', fontsize=32, fontweight='bold', color=COLORS['navy'])

ax.text(5, 4.0, r"where  $\alpha = \frac{1}{1-\beta}$",
        ha='center', va='center', fontsize=16, color=COLORS['medium'])

definitions = [
    (1.5, 2.8, r"$U$", "Effective Capability", "What the system achieves"),
    (4.0, 2.8, r"$I$", "Base Potential", "Frozen disorder"),
    (6.5, 2.8, r"$R$", "Recursive Depth", "Self-referential cycles"),
    (9.0, 2.8, r"$\alpha$", "Scaling Exponent", "Compounding efficiency"),
]

for x, y, symbol, name, desc in definitions:
    circle = plt.Circle((x, y), 0.35, fill=True, facecolor=COLORS['bg_light'],
                         edgecolor=COLORS['navy'], linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['navy'])
    ax.text(x, y - 0.6, name, ha='center', va='top', fontsize=9, fontweight='bold', color=COLORS['dark'])
    ax.text(x, y - 0.95, desc, ha='center', va='top', fontsize=8, color=COLORS['light'])

box = FancyBboxPatch((3.0, 0.3), 4.0, 1.2, boxstyle="round,pad=0.05,rounding_size=0.1",
                      facecolor=COLORS['bg_light'], edgecolor=COLORS['border'], linewidth=1)
ax.add_patch(box)
ax.text(5, 1.1, r"$\beta$ = self-referential coupling", ha='center', va='center',
        fontsize=10, fontweight='bold', color=COLORS['dark'])
ax.text(5, 0.7, "How much prior work helps the next step", ha='center', va='center',
        fontsize=9, color=COLORS['medium'])

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "fig1_equation.png")
plt.savefig(output_path, facecolor='white', edgecolor='none')
plt.close()
print(f"Generated: {output_path}")
