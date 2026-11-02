import json
import pathlib
import random
import sys

import anthropic
import pymupdf4llm

MODEL = "claude-haiku-5-5"
client = anthropic.Anthropic()

PROMPT = """Below is one page from the document "{file}".
Write ONE question that a student might realistically ask and that this page answers, plus the short correct answer.
Rules:
- The answer must come from this page only.
- Phrase the question the way a person would ask it, in your own words. Do not copy phrases from the page.
- Skip trivia such as page headers, form numbers on their own, or contact details.
Reply with JSON only: {{"q": "...", "answer": "..."}}
If the page has nothing worth asking about, reply {{"skip": true}}

<page>
{text}
</page>"""

def ask(file, text):
    response = client.messages.create(
        model=MODEL, max_tokens=300,
        messages=[{"role": "user", "content": PROMPT.format(file=file, text=text[:6000])}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    try:
        return json.loads(raw[raw.index("{"): raw.rindex("}") + 1]) 
    except ValueError:
        return {"skip": True}
    
def main(folder,per_file):
    random.seed(0)
    with open("evals/questions_draft.jsonl", "w") as out:
        for pdf in sorted(pathlib.Path(folder).glob("*.pdf")):
            pages = [p for p in pymupdf4llm.to_markdown(str(pdf), page_chunks=True) if len(p["text"]) > 500]
            for page in random.sample(pages, min(per_file, len(pages))):
                data = ask(pdf.name, page["text"])
                if data.get("skip") or "q" not in data:
                    continue
                case = {"q": data["q"], "file": pdf.name,
                        "page": page["metadata"]["page_number"], "answer": data["answer"]}
                out.write(json.dumps(case) + "\n")
                print(f"{pdf.name} p{case['page']}: {case['q']}")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)

