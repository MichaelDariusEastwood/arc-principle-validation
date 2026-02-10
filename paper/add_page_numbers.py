#!/usr/bin/env python3
"""
Add Page Numbers to PDF
=======================

Adds professional page numbers and running footers to the Paper III PDF.

Author: Michael Darius Eastwood
"""

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import sys


def add_page_numbers(input_pdf, output_pdf):
    """Add page numbers and running footer to PDF."""

    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    num_pages = len(reader.pages)

    for page_num, page in enumerate(reader.pages):
        # Create overlay with page number
        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=A4)

        width, height = A4  # 595.27, 841.89 points

        # Page number at bottom center
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.4, 0.4, 0.4)  # Gray
        c.drawCentredString(width / 2, 1.5 * cm, str(page_num + 1))

        # Running footer on left (skip first page)
        if page_num > 0:
            c.setFont("Helvetica", 8)
            c.setFillColorRGB(0.6, 0.6, 0.6)  # Light gray
            c.drawString(2 * cm, 1.5 * cm, "Eastwood's ARC Principle | White Paper III")

            # Date on right
            c.drawRightString(width - 2 * cm, 1.5 * cm, "9 February 2026")

        c.save()

        # Move to beginning of BytesIO buffer
        packet.seek(0)

        # Read the overlay
        overlay_reader = PdfReader(packet)
        overlay_page = overlay_reader.pages[0]

        # Merge overlay onto page
        page.merge_page(overlay_page)
        writer.add_page(page)

    # Write output
    with open(output_pdf, 'wb') as f:
        writer.write(f)

    print(f"Added page numbers to {num_pages} pages")
    print(f"Output: {output_pdf}")


if __name__ == '__main__':
    input_file = 'EASTWOOD-ARC-PRINCIPLE-PAPER-III-v6.1.pdf'
    output_file = 'EASTWOOD-ARC-PRINCIPLE-PAPER-III-v6.1-FINAL.pdf'

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    add_page_numbers(input_file, output_file)
