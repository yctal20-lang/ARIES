# -*- coding: utf-8 -*-
"""One-off: copy presentation + speech from Desktop into repo."""
import os
import shutil
from pathlib import Path

DESKTOP = Path(r"c:\Users\Asus\Desktop")
REPO = Path(r"c:\Users\Asus\Desktop\CURSOR")
ARIES = REPO / "ARIES"
DOCS = REPO / "docs"

src_pdf = DESKTOP / "\u00abA.R.I.E.S\u00bb.pdf"  # «A.R.I.E.S».pdf
src_docx = DESKTOP / "_\u0420\u0415\u0427\u042c (\u0421\u0422\u0420\u0423\u041a\u0422\u0423\u0420\u0418\u0420\u041e\u0412\u0410\u041d\u041d\u0410\u042f, \u0411\u0415\u0417 \u0418\u0417\u041c\u0415\u041d\u0415\u041d\u0418\u0419 \u0422\u0415\u041a\u0421\u0422\u0410).docx"

def main() -> None:
    for p in (src_pdf, src_docx):
        if not p.is_file():
            raise SystemExit(f"Missing source: {p}")

    DOCS.mkdir(parents=True, exist_ok=True)
    dst_pdf_docs = DOCS / src_pdf.name
    dst_docx_docs = DOCS / src_docx.name
    shutil.copy2(src_pdf, dst_pdf_docs)
    shutil.copy2(src_docx, dst_docx_docs)

    ARIES.mkdir(parents=True, exist_ok=True)
    # Replace any prior PDF in ARIES (angle brackets vs guillemets naming)
    for old in ARIES.glob("*.pdf"):
        old.unlink()
    shutil.copy2(src_pdf, ARIES / src_pdf.name)

    for old in ARIES.glob("*.docx"):
        old.unlink()
    shutil.copy2(src_docx, ARIES / src_docx.name)

    print("OK:", dst_pdf_docs, dst_docx_docs, ARIES / src_pdf.name, ARIES / src_docx.name)

if __name__ == "__main__":
    main()
