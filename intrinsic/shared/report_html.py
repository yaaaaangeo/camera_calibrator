"""
shared.report_html
==================

HTML 리포트의 공용 틀 (스타일·페이지·표) - Camera Intrinsic 리포트(export/report.py)와
CAMERA–LiDAR 카메라별 리포트(report.html)가 같은 모양으로 나온다.
"""

from __future__ import annotations

import html

CSS = """
body { font-family: -apple-system, "Segoe UI", "Malgun Gothic", sans-serif; margin: 0; padding: 32px;
       color: #1a1a1a; background: #fff; max-width: 900px; margin-left:auto; margin-right:auto; }
h1 { font-size: 22px; border-bottom: 3px solid #1a1a1a; padding-bottom: 8px; }
h2 { font-size: 17px; margin-top: 36px; border-left: 5px solid #455a64; padding-left: 10px; }
h4 { font-size: 14px; margin-top: 18px; color: #444; }
.meta { color: #666; font-size: 13px; margin-bottom: 20px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
table.kv th { text-align: left; width: 220px; background: #f5f5f5; }
table th, table td { border: 1px solid #ddd; padding: 6px 10px; }
table.compare th, table.compare td { text-align: center; }
table.compare thead th { background: #37474f; color: #fff; }
table.compare tbody th { text-align: left; background: #f5f5f5; }
.bars { margin: 12px 0; }
.bar-row { display: flex; align-items: center; margin: 4px 0; font-size: 13px; }
.bar-label { width: 190px; }
.bar-track { flex: 1; background: #eee; border-radius: 3px; height: 14px; overflow: hidden; }
.bar-fill { background: #43a047; height: 100%; }
.bar-pct { width: 44px; text-align: right; }
.grade-box { display: inline-block; padding: 14px 22px; border-radius: 6px; color: #fff;
             font-size: 18px; font-weight: bold; margin-top: 10px; }
.stars { font-size: 20px; letter-spacing: 2px; }
@media print { body { padding: 0; } h2 { break-after: avoid; } table { break-inside: avoid; } }
"""


def esc(s: object) -> str:
    return html.escape(str(s))


def page(title: str, body: str, intro: str = "") -> str:
    """완성된 HTML 문서 한 장."""
    return ("<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'>"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body>{intro}{body}</body></html>")


def kv_table(rows) -> str:
    """[(이름, 값)] -> 2열 표."""
    cells = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows)
    return f"<table class='kv'>{cells}</table>"


def table(headers, rows) -> str:
    """머리글 + 줄들 -> 비교 표."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table class='compare'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
