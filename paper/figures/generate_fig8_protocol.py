#!/usr/bin/env python3
"""
Generate Figure 8: Global Scaling Challenge Measurement Protocol
================================================================

Creates publication-quality figure showing the 6-step standardised
measurement protocol for testing the ARC Principle.

Author: Michael Darius Eastwood
License: MIT
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def set_nature_style():
    """Configure matplotlib for Nature-style figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.linewidth': 1.0,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def generate_protocol_figure(output_file='fig8_protocol.png'):
    """Generate the measurement protocol figure."""

    set_nature_style()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.6, 'GLOBAL SCALING CHALLENGE',
            ha='center', va='top', fontsize=16, fontweight='bold',
            color='#1a1a1a')
    ax.text(5, 9.2, 'Standardised Measurement Protocol for Testing the ARC Principle',
            ha='center', va='top', fontsize=12, color='#666666', style='italic')

    # Colors
    step_color = '#1f77b4'  # Blue for steps
    arrow_color = '#666666'
    box_bg = '#f5f8ff'
    critical_color = '#c41230'  # Red for critical step

    # Step positions (y decreases as step number increases)
    steps = [
        {
            'num': '1',
            'title': 'Define Recursive Depth (R)',
            'desc': 'Measure depth in comparable units\n(tokens, iterations, cycles)',
            'y': 8.2,
            'critical': False
        },
        {
            'num': '2',
            'title': 'Measure Capability (U)',
            'desc': 'Standardised benchmarks with\ninter-rater reliability > 0.85',
            'y': 7.0,
            'critical': False
        },
        {
            'num': '3',
            'title': 'Collect Multi-Scale Data',
            'desc': 'Minimum 5 data points spanning\nat least one order of magnitude',
            'y': 5.8,
            'critical': False
        },
        {
            'num': '4',
            'title': 'Fit Multiple Functional Forms',
            'desc': 'Compare power law, exponential,\nand logarithmic fits (AIC/BIC)',
            'y': 4.6,
            'critical': False
        },
        {
            'num': '5',
            'title': 'Report Alpha with Uncertainty',
            'desc': '95% confidence intervals required',
            'y': 3.4,
            'critical': False
        },
        {
            'num': '5b',
            'title': 'Test Beta-Derivation Independently',
            'desc': 'Verify that measured alpha satisfies\nalpha = 1/(1-beta) within error',
            'y': 2.2,
            'critical': True
        },
        {
            'num': '6',
            'title': 'Submit to Repository',
            'desc': 'Data and analysis code available at:\ngithub.com/MichaelDariusEastwood/arc-scaling-challenge',
            'y': 1.0,
            'critical': False
        },
    ]

    box_width = 7.5
    box_height = 0.9

    for i, step in enumerate(steps):
        y = step['y']
        color = critical_color if step['critical'] else step_color
        bg = '#fff5f5' if step['critical'] else box_bg

        # Draw box
        box = FancyBboxPatch(
            (1.25, y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=bg,
            edgecolor=color,
            linewidth=2 if step['critical'] else 1.5,
            zorder=2
        )
        ax.add_patch(box)

        # Step number circle
        circle = plt.Circle((1.0, y), 0.35,
                           facecolor=color, edgecolor='white',
                           linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(1.0, y, step['num'],
               ha='center', va='center', fontsize=10, fontweight='bold',
               color='white', zorder=4)

        # Title and description
        ax.text(1.6, y + 0.15, step['title'],
               ha='left', va='center', fontsize=11, fontweight='bold',
               color='#1a1a1a', zorder=4)
        ax.text(1.6, y - 0.25, step['desc'],
               ha='left', va='center', fontsize=9,
               color='#666666', zorder=4)

        # Arrow to next step
        if i < len(steps) - 1:
            next_y = steps[i + 1]['y']
            arrow = FancyArrowPatch(
                (5, y - box_height/2 - 0.05),
                (5, next_y + box_height/2 + 0.05),
                arrowstyle='->,head_length=6,head_width=4',
                color=arrow_color,
                linewidth=1.5,
                zorder=1
            )
            ax.add_patch(arrow)

    # Key requirements box on the right
    req_x = 9.2
    req_y = 5.5
    ax.text(req_x, req_y + 2.0, 'KEY REQUIREMENTS',
           ha='center', va='center', fontsize=10, fontweight='bold',
           color='#1a1a1a')

    requirements = [
        'Multi-scale data (R spans 10x)',
        'Compare multiple models',
        'Independent beta test',
        '95% confidence intervals',
        'Negative results welcome',
    ]

    for i, req in enumerate(requirements):
        ax.text(req_x, req_y + 1.3 - i * 0.45, f'  {req}',
               ha='center', va='center', fontsize=8,
               color='#333333')
        # Checkmark
        ax.text(req_x - 1.3, req_y + 1.3 - i * 0.45, '',
               ha='center', va='center', fontsize=10,
               color='#28a745')

    # Footer with repository link
    ax.text(5, 0.15, 'Analysis toolkit: github.com/MichaelDariusEastwood/arc-scaling-challenge',
           ha='center', va='center', fontsize=9, color='#0066cc',
           style='italic')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"Saved: {output_file}")

    return fig


if __name__ == '__main__':
    generate_protocol_figure()
