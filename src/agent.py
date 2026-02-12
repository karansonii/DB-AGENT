```python
# agent.py
# DB Intelligence Agent (Question Mode + Strict SQL Mode)

import json
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.utils import (
    get_llm,
    get_postgres_engine,
    get_qdrant_client,
    get_embedding_model,
    load_config,
    create_qdrant_collection
)

from qdrant_client.http.models import Filter, FieldCondition, MatchValue


# ─────────────────────────────────────────────
# TOOL 1 — Execute SQL safely
# ─────────────────────────────────────────────
def postgres_query(sql: str, environment: str = "dev") -> str:

    engine = get_postgres_engine(environment)

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql)).fetchall()
            rows = [dict(row._mapping) for row in result]
            return json.dumps(rows, default=str)

    except Exception as e:
        return f"DB error: {str(e)}"


# ─────────────────────────────────────────────
# TOOL 2 — Qdrant Search (unchanged)
# ─────────────────────────────────────────────
def qdrant_search(query_text: str, environment: str = "dev", limit: int = 5):

    client = get_qdrant_client(environment)
    model = get_embedding_model()
    config = load_config(environment)

    collection = config.get("qdrant", {}).get("collection", "knowledge")
    vector_size = config.get("vector_size", 1024)
    project = config["project_name"]

    create_qdrant_collection(client, collection, vector_size)

    embedding = model.encode(query_text).tolist()

    filter_ = Filter(
        must=[
            FieldCondition(
                key="project_name",
                match=MatchValue(value=project)
            ),
        ]
    )

    result = client.query_points(
        collection_name=collection,
        query=embedding,
        query_filter=filter_,
        limit=limit,
    )

    return [point.payload for point in result.points]


# ─────────────────────────────────────────────
# STRICT SQL MODE (Query Box)
# ─────────────────────────────────────────────
def run_sql_query(user_sql: str, env: str = "dev") -> str:

    engine = get_postgres_engine(env)
    sql = user_sql.strip()

    # 1️⃣ Validate using EXPLAIN (no execution yet)
    try:
        with engine.connect() as conn:
            conn.execute(text(f"EXPLAIN {sql}"))
    except Exception as e:
        return (
            f"❌ Syntax Error Detected\n\n"
            f"Your SQL:\n{sql}\n\n"
            f"Error:\n{str(e)}"
        )

    # 2️⃣ Execute
    result = postgres_query(sql, env)

    if not result or result == "[]":
        return (
            f"✅ Query Valid (No Syntax Errors)\n\n"
            f"Executed SQL:\n{sql}\n\n"
            f"No rows returned."
        )

    return (
        f"✅ Query Valid\n\n"
        f"Executed SQL:\n{sql}\n\n"
        f"Result:\n{result}"
    )


# ─────────────────────────────────────────────
# QUESTION MODE (Ask Box)
# ─────────────────────────────────────────────
def run_question_query(user_question: str, env: str = "dev") -> str:

    llm = get_llm()
    engine = get_postgres_engine(env)

    # 1️⃣ Fetch live schema from information_schema
    with engine.connect() as conn:
        schema_data = conn.execute(text("""
            SELECT table_schema, table_name, column_name
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
        """)).fetchall()

    schema_list = [
        f"{row.table_schema}.{row.table_name}.{row.column_name}"
        for row in schema_data
    ]

    # 2️⃣ Prompt LLM to generate SQL
    sql_prompt = f"""
You are a PostgreSQL SQL generator.

STRICT RULES:
- Use ONLY available tables.
- Do NOT hallucinate.
- If asking about tables -> query information_schema.
- Return ONLY SQL.
- No markdown.
- No explanations.

User Question:
{user_question}

Available Schema:
{schema_list}
"""

    sql_response = llm.invoke(sql_prompt)
    generated_sql = sql_response.content.strip()

    # 3️⃣ Validate SQL
    try:
        with engine.connect() as conn:
            conn.execute(text(f"EXPLAIN {generated_sql}"))
    except Exception as e:
        return (
            f"❌ Generated SQL Invalid\n\n"
            f"Generated SQL:\n{generated_sql}\n\n"
            f"Error:\n{str(e)}"
        )

    # 4️⃣ Execute
    result = postgres_query(generated_sql, env)

    if not result or result == "[]":
        return (
            f"Generated SQL:\n{generated_sql}\n\n"
            f"No rows returned."
        )

    return (
        f"Generated SQL:\n{generated_sql}\n\n"
        f"Result:\n{result}"
    )
```
