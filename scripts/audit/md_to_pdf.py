"""Render a Markdown report to a publication-style PDF with ReportLab.

Handles the subset of Markdown the audit reports actually use: ATX headings,
pipe tables, blockquotes, bullet lists, horizontal rules, and inline
**bold** / *italic* / `code`. Anything else passes through as body text.

Usage:  python scripts/audit/md_to_pdf.py <input.md> [output.pdf]
"""
import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (HRFlowable, KeepTogether, ListFlowable, ListItem,
                                 PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                 TableStyle)

INK = colors.HexColor("#1a1d23")
MUTED = colors.HexColor("#5b6169")
ACCENT = colors.HexColor("#8a1538")        # matches the project's deck accent
RULE = colors.HexColor("#d6d2c8")
CODE_BG = colors.HexColor("#f2f0eb")
QUOTE_BG = colors.HexColor("#f7f3e8")


def styles():
    ss = getSampleStyleSheet()
    # No `leading` here -- each style sets its own, and duplicating it in the
    # shared dict collides with the explicit keyword.
    base = dict(fontName="Helvetica", textColor=INK)
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                                 fontSize=19, leading=23, textColor=INK,
                                 spaceAfter=2, alignment=0),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13.2, leading=16,
                              textColor=ACCENT, spaceBefore=15, spaceAfter=5),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11, leading=14,
                              textColor=INK, spaceBefore=10, spaceAfter=3),
        "body": ParagraphStyle("b", fontSize=9.4, leading=13.6, alignment=TA_JUSTIFY,
                                spaceAfter=6, **base),
        "src": ParagraphStyle("s", fontName="Helvetica-Oblique", fontSize=8.2,
                               leading=11, textColor=MUTED, spaceAfter=9),
        "quote": ParagraphStyle("q", fontSize=9.2, leading=13, textColor=INK,
                                 leftIndent=9, rightIndent=7, spaceBefore=3,
                                 spaceAfter=8, borderPadding=7,
                                 backColor=QUOTE_BG, borderColor=ACCENT, borderWidth=0,
                                 fontName="Helvetica"),
        "li": ParagraphStyle("li", fontSize=9.4, leading=13.2, spaceAfter=3, **base),
        "cellh": ParagraphStyle("ch", fontName="Helvetica-Bold", fontSize=8.3,
                                 leading=10.5, textColor=colors.white),
        "cell": ParagraphStyle("c", fontName="Helvetica", fontSize=8.3, leading=10.5,
                                textColor=INK),
    }


def inline(t: str) -> str:
    """Markdown inline -> ReportLab mini-HTML, escaping first."""
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`",
               r'<font face="Courier" size="8.4" backColor="#f2f0eb">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    t = t.replace("--", "&#8211;")
    return t


def split_row(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build(md: str, out: Path, subtitle: str = ""):
    S = styles()
    story, lines, i = [], md.splitlines(), 0
    first_h1 = True

    while i < len(lines):
        ln = lines[i]

        # --- table
        if ln.strip().startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header = split_row(ln)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            data = [[Paragraph(inline(c), S["cellh"]) for c in header]]
            for r in rows:
                r += [""] * (len(header) - len(r))
                data.append([Paragraph(inline(c), S["cell"]) for c in r[:len(header)]])
            avail = A4[0] - 36 * mm
            tbl = Table(data, colWidths=[avail / len(header)] * len(header),
                        repeatRows=1, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#faf8f4")]),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.extend([Spacer(1, 3), tbl, Spacer(1, 8)])
            continue

        s = ln.strip()

        if not s:
            i += 1
            continue

        if s.startswith("# "):
            if first_h1:
                story.append(Paragraph(inline(s[2:]), S["title"]))
                if subtitle:
                    story.append(Paragraph(subtitle, S["src"]))
                story.append(HRFlowable(width="100%", thickness=1.1, color=ACCENT,
                                         spaceBefore=3, spaceAfter=9))
                first_h1 = False
            else:
                story.append(Paragraph(inline(s[2:]), S["h2"]))
        elif s.startswith("## "):
            story.append(Paragraph(inline(s[3:]), S["h2"]))
            story.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                     spaceBefore=1, spaceAfter=5))
        elif s.startswith("### "):
            story.append(Paragraph(inline(s[4:]), S["h3"]))
        elif s.startswith(">"):
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            story.append(Paragraph(inline(" ".join(block)), S["quote"]))
            continue
        elif re.match(r"^[-*+] ", s) or re.match(r"^\d+\. ", s):
            items, ordered = [], bool(re.match(r"^\d+\. ", s))
            while i < len(lines):
                c = lines[i].strip()
                if re.match(r"^[-*+] ", c):
                    items.append(c[2:])
                elif re.match(r"^\d+\. ", c):
                    items.append(re.sub(r"^\d+\.\s*", "", c))
                elif c == "" and items:
                    break
                else:
                    break
                i += 1
            story.append(ListFlowable(
                [ListItem(Paragraph(inline(x), S["li"]), leftIndent=13) for x in items],
                bulletType="1" if ordered else "bullet",
                bulletFontSize=8, start="1" if ordered else None,
                leftIndent=12, bulletColor=ACCENT))
            story.append(Spacer(1, 4))
            continue
        elif set(s) <= set("-*_") and len(s) >= 3:
            story.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                     spaceBefore=5, spaceAfter=5))
        elif s.startswith("*Source:") or s.startswith("*Sources:"):
            story.append(Paragraph(inline(s.strip("*")), S["src"]))
        else:
            story.append(Paragraph(inline(s), S["body"]))
        i += 1

    def furniture(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 12 * mm, out.stem.replace("_", " "))
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"page {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title=out.stem.replace("_", " "),
        author="Occlusion-Aware Camera-Radar Fusion project")
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "MANUSCRIPT_RESULTS.md")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
    sub = ("Confidence-Driven Occlusion-Aware Camera-Radar Fusion & Hidden Hazard "
           "Prediction for ADAS - recomputed from raw data by scripts/audit/")
    build(src.read_text(encoding="utf-8"), out, sub)
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
