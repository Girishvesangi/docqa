import pathlib
import sys

from app.rag import ingest

folder = pathlib.Path(sys.argv[1])
for pdf in sorted(folder.glob("*.pdf")):
    if "--context" in sys.argv: 
        pages = ingest(str(pdf), pdf.name, use_context=True)
    else:
        pages = ingest(str(pdf), pdf.name)
    print(f"{pdf.name}: {pages} pages")