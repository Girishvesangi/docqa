import csv
import datetime
import json
import sys

from app.rag import MODEL, answer, client, retrieve

JUDGE = """You grade a document Q&A system. You get a question, the expected answer, and the
system's answer. Reply PASS if the system's answer agrees with the expected answer (extra correct
detail is fine; for out-of-scope questions it must say it couldn't find the answer).
Otherwise reply FAIL. Reply with one word."""


def judge(case, got):
    response = client.messages.create(
        model=MODEL, max_tokens=1024, system=JUDGE,
        messages=[{"role": "user", "content":
                   f"Question: {case['q']}\nExpected: {case['answer']}\nSystem answer: {got}"}],
    )
    verdict = "".join(b.text for b in response.content if b.type == "text")
    return verdict.strip().upper().startswith("PASS")

def main(label):
    with open("evals/questions.jsonl") as f:
        cases=[json.loads(line) for line in f if line.strip()]
    in_scope=[c for c in cases if c.get("file")]
    hits=correct=0
    for c in cases:
        if c.get("file"):
            found=retrieve(c["q"],k=5)
            if any(h["source"] == c["file"] and h["page"] == c["page"] for h in found):
                hits+=1
            else:
                print("RETRIEVAL MISS:", c["q"])
        got,_=answer(c["q"])
        if judge(c, got):
            correct+=1
        else:
            print("WRONG ANSWER:", c["q"], "->", got[:150].replace("\n", " "))
    hit_rate=hits/max(len(in_scope),1)
    accuracy = correct / len(cases)
    print(f"\n{label}: hit@5 = {hit_rate:.0%}   answer accuracy = {accuracy:.0%}")
    with open("evals/results.csv", "a", newline="") as f:
        csv.writer(f).writerow([datetime.date.today(), label, f"{hit_rate:.3f}", f"{accuracy:.3f}"])

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")