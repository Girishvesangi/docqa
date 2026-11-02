import os
import re

import anthropic
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import pymupdf4llm

DB_URL=os.environ.get("DATABASE_URL","postgresql://postgres:pw@localhost:5432/postgres")
reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embedder=SentenceTransformer("all-MiniLM-L6-v2")
client = anthropic.Anthropic()
MODEL="claude-haiku-5-5"

with psycopg.connect(DB_URL,autocommit=True) as setup:

    setup.execute("CREATE EXTENSION IF NOT EXISTS vector")
    setup.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section text")
    setup.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context text")
    setup.execute(
    """
    CREATE TABLE IF NOT EXISTS chunks (
    id bigserial PRIMARY KEY,
    source text, page int, content text,
    embedding vector(384)
    )
    """
    )
    setup.execute("CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
    setup.execute("""
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(context, '') || ' ' || coalesce(section, '') || ' ' || content)
        ) STORED""")
    setup.execute("CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv)")


pool=ConnectionPool(
    DB_URL,
    min_size=1,
    max_size=5,
    kwargs={"autocommit":True},
    configure=register_vector,
    check=ConnectionPool.check_connection,
    open=True,
)
SYSTEM = """Answer the question using only the numbered sources provided.
Cite sources inline like [1] or [2][3]. If the sources do not contain the answer,
say "I couldn't find that in the documents." Never make up information."""

HEADING=re.compile(r"^(#{1,6})\s+(.*)")
TAGS = re.compile(r"</?(u|mark|sup|sub|br|span)\b[^>]*>", re.I)
COLUMNS = "id, source, page, section, content"

def clean(md):
    md = TAGS.sub(" ", md).replace("**", "").replace("\x00", "")
    return re.sub(r"[ \t]{2,}", " ", md)

def split_long(text, size=1000, overlap=150):
    pieces,start=[],0
    while start<len(text):
        pieces.append(text[start:start+size])
        start+=size-overlap
    return pieces

def chunk_markdown(md, heading="",max_chars=1000):
    sections, buf = [], []
    for line in md.splitlines():
        m = HEADING.match(line)
        if m:
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(2).strip(" *"), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))

    chunks = []
    for section, text in sections:
        current = ""
        for para in (p.strip() for p in text.split("\n\n")):
            if not para:
                continue
            if len(para) > max_chars:
                if current:
                    chunks.append((section, current))
                    current = ""
                chunks.extend((section, piece) for piece in split_long(para, max_chars))
                continue
            if current and len(current) + len(para) + 2 > max_chars:
                chunks.append((section, current))
                current = ""
            current = f"{current}\n\n{para}" if current else para
        if current:
            chunks.append((section, current))
    return chunks, heading

def contextualize(doc_text, chunk_text):
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": f"<document>\n{doc_text}\n</document>",
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content":
                   f"Here is a chunk from the document above:\n<chunk>\n{chunk_text}\n</chunk>\n"
                   "Write 1-2 sentences that situate this chunk within the document, "
                   "to improve search retrieval. Reply with only those sentences."}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()



def ingest(path,name, use_context=False):
    """Index one PDF. reuploading the same file name replaces it's older chunks"""
    pages=pymupdf4llm.to_markdown(path,page_chunks=True)
    rows,heading=[],""
    for page in pages:
        text=page["text"].replace("\x00","")
        chunks,heading= chunk_markdown(text,heading)
        for section, content in chunks:
            rows.append({"source": name, "page": page["metadata"]["page_number"],
                         "section": section, "content": content, "context": None})

    if not rows:
        return 0
    if use_context:
        doc_text = "\n\n".join(p["text"] for p in pages)[:400_000] 
        for r in rows:
            r["context"] = contextualize(doc_text, r["content"])
    texts = [f"{r['context'] or ''}\n{r['source']} > {r['section']}\n{r['content']}" for r in rows]
    vectors= embedder.encode(texts,batch_size=32)
    with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source=%s",(name,))
        cur.executemany(
            "INSERT INTO chunks (source,page,section,context,content,embedding ) VALUES (%s, %s,%s,%s, %s, %s)",
            [(r["source"], r["page"], r["section"], r["context"], r["content"], v)
             for r, v in zip(rows, vectors)],
        )
    return len(pages)





def vector_search(query, k):
    q = embedder.encode(query)
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT {COLUMNS} FROM chunks ORDER BY embedding <=> %s LIMIT %s", (q, k))
        return cur.fetchall()


def keyword_search(query, k):
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            # OR the words together: a question rarely contains every word of the answer
            f"SELECT {COLUMNS} FROM chunks, "
            "to_tsquery('english', replace(plainto_tsquery('english', %s)::text, '&', '|')) q "
            "WHERE tsv @@ q ORDER BY ts_rank_cd(tsv, q) DESC LIMIT %s",
            (query, k),
        )
        return cur.fetchall()


def rrf(result_lists, k=60):
    """Reciprocal Rank Fusion: merge ranked lists, rewarding rows ranked high in any list."""
    scores, rows = {}, {}
    for results in result_lists:
        for rank, row in enumerate(results):
            scores[row["id"]] = scores.get(row["id"], 0) + 1 / (k + rank + 1)
            rows[row["id"]] = row
    return [rows[i] for i in sorted(scores, key=scores.get, reverse=True)]


def hybrid_search(queries, k=20):
    lists = []
    for q in queries:
        lists += [vector_search(q, k), keyword_search(q, k)]
    return rrf(lists)[:k]

def retrieve(question, k=5, candidates=20, expand=False):
    queries = [question]
    rows = hybrid_search(queries, candidates)
    if not rows:
        return []
    scores = reranker.predict([(question, f"{r['section']}\n{r['content']}") for r in rows])
    for r, s in zip(rows, scores):
        r["score"] = float(s)
    return sorted(rows, key=lambda r: r["score"], reverse=True)[:k]


def retrieve(question, k=5):
    return rrf([vector_search(question, 20), keyword_search(question, 20)])[:k]

def answer(question):
    hits=retrieve(question)
    if not hits:
        return "I couldn't find that in the documents.", []
    sources = "\n\n".join(
    f'<source id="{i}" file="{h["source"]}" page="{h["page"]}">\n{h["content"]}\n</source>'
    for i, h in enumerate(hits, start=1)
)
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"{sources}\n\nQuestion: {question}"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, [(i, h["source"], h["page"]) for i,h in enumerate(hits, start=1)]









