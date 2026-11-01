import os

import anthropic
import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DB_URL=os.environ.get("DATABASE_URL","postgresql://postgres:pw@localhost:5432/postgres")

embedder=SentenceTransformer("all-MiniLM-L6-v2")
client = anthropic.Anthropic()

with psycopg.connect(DB_URL,autocommit=True) as setup:

    setup.execute("CREATE EXTENSION IF NOT EXISTS vector")
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

def chunk(text, size=800, overlap=150):
    pieces, start=[],0
    while start<len(text):
        pieces.append(text[start:start+size])
        start+=size-overlap
    return pieces

def ingest(path,name):
    """Index one PDF. reuploading the same file name replaces it's older chunks"""
    reader=PdfReader(path)
    rows=[]
    for page_num, page in enumerate(reader.pages, start=1):
        text=(page.extract_text() or "").replace("\x00", "")
        for piece in chunk(text):
            rows.append((name,page_num,piece))
    if not rows:
        return 0
    vectors= embedder.encode([r[2] for r in rows])
    with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source=%s",(name,))
        cur.executemany(
            "INSERT INTO chunks (source,page,content,embedding ) VALUES (%s, %s, %s, %s)",
            [(s, p, c, v) for (s, p, c), v in zip(rows, vectors)],
        )
    return len(reader.pages)


def retrieve(question,k=5):
    q=embedder.encode(question)
    with pool.connection() as conn:
        return conn.execute(
            "SELECT source, page, content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (q, k),
        ).fetchall()

def answer(question):
    hits=retrieve(question)
    if not hits:
        return "I couldn't find that in the documents.", []
    sources="\n\n".join(
        f'<source id="{i}" file="{src}" page="{page}">\n{text}\n</source>'
        for i, (src,page,text) in enumerate(hits,start=1)
    )
    
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"{sources}\n\nQuestion: {question}"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, [(i, src, page) for i, (src, page, _) in enumerate(hits, start=1)]



