```python
import streamlit as st
from src.utils import (
    get_postgres_engine,
    get_qdrant_client,
    get_embedding_model,
    create_qdrant_collection,
    load_config
)

from sqlalchemy import text
from qdrant_client.http.models import PointStruct
import sqlparse
import pandas as pd
import uuid
import re


# ─────────────────────────────────────────────
# PostgreSQL table creation SQL
# ─────────────────────────────────────────────
PG_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS tables_metadata (
    id SERIAL PRIMARY KEY,
    project_name VARCHAR(255),
    database_name VARCHAR(255),
    environment VARCHAR(50),
    table_schema VARCHAR(255),
    table_name VARCHAR(255),
    column_name VARCHAR(255),
    data_type VARCHAR(50),
    nullable BOOLEAN,
    default_value TEXT,
    UNIQUE(project_name, database_name, environment, table_schema, table_name, column_name)
);

CREATE TABLE IF NOT EXISTS queries (
    id SERIAL PRIMARY KEY,
    query_name VARCHAR(255) UNIQUE,
    normalized_template TEXT,
    environment VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS execution_evidence (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES queries(id),
    explain_output JSONB,
    anti_patterns TEXT[]
);
"""



# ─────────────────────────────────────────────
# Create Tables
# ─────────────────────────────────────────────
def create_pg_tables(env='dev'):
    engine = get_postgres_engine(env)

    with engine.connect() as conn:
        # 🔎 Verify which DB we are connected to
        current_db = conn.execute(text("SELECT current_database()")).scalar()
        print("CONNECTED TO DATABASE:", current_db)

        # 🔎 Verify schema
        conn.execute(text("SET search_path TO public"))

        # 🧹 Optional: drop old broken tables (safe reset)
        conn.execute(text("DROP TABLE IF EXISTS execution_evidence CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS queries CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS tables_metadata CASCADE"))

        # 🏗 Recreate tables
        conn.execute(text(PG_TABLES_SQL))
        conn.commit()

    print("Postgres metadata tables created successfully.")



# ─────────────────────────────────────────────
# Metadata Ingestion (AUTO-DETECT SCHEMA)
# ─────────────────────────────────────────────
def ingest_metadata_from_postgres(env='dev'):

    engine = get_postgres_engine(env)
    config = load_config(env)

    project = config.get("project_name", "eGP")
    db_name = config["postgres"]["db"]

    with engine.connect() as conn:

        df = pd.read_sql("""
            SELECT table_schema,
                   table_name,
                   column_name,
                   data_type,
                   is_nullable='YES' AS nullable,
                   column_default
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
        """, conn)

        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO tables_metadata
                (project_name,database_name,environment,
                 table_schema,table_name,column_name,
                 data_type,nullable,default_value)
                VALUES (:p,:d,:e,:s,:t,:c,:dt,:n,:def)
                ON CONFLICT DO NOTHING
            """), {
                "p": project,
                "d": db_name,
                "e": env,
                "s": row.table_schema,
                "t": row.table_name,
                "c": row.column_name,
                "dt": row.data_type,
                "n": row.nullable,
                "def": row.column_default
            })

        conn.commit()

    st.success("✅ Metadata ingested from all user schemas")


# ─────────────────────────────────────────────
# Query Ingestion
# ─────────────────────────────────────────────
def ingest_queries(queries_list, env='dev'):

    engine = get_postgres_engine(env)

    with engine.connect() as conn:
        for q in queries_list:

            normalized = sqlparse.format(
                q["template"],
                strip_comments=True,
                keyword_case="upper"
            )

            # Remove schema prefix automatically
            normalized = re.sub(r'\b\w+\.', '', normalized)

            # Replace values with placeholders
            normalized = re.sub(r"\d+", "?", normalized)
            normalized = re.sub(r"'[^']*'", "?", normalized)

            conn.execute(text("""
                INSERT INTO queries
                (query_name,normalized_template,environment)
                VALUES (:n,:t,:e)
                ON CONFLICT DO NOTHING
            """), {
                "n": q["name"],
                "t": normalized,
                "e": env
            })

        conn.commit()


# ─────────────────────────────────────────────
# Evidence Generation (SAFE VERSION)
# ─────────────────────────────────────────────
def generate_evidence(query_id, env='dev'):

    engine = get_postgres_engine(env)

    with engine.connect() as conn:

        # Automatically pick schema where tenders exists
        schema_row = conn.execute(text("""
            SELECT table_schema
            FROM information_schema.tables
            WHERE table_name='tenders'
            LIMIT 1
        """)).fetchone()

        if not schema_row:
            st.warning("⚠️ Table 'tenders' not found in any schema")
            return

        schema_name = schema_row[0]

        # Set schema context
        conn.execute(text(f"SET search_path TO {schema_name}"))

        res = conn.execute(
            text("SELECT normalized_template FROM queries WHERE id=:id"),
            {"id": query_id}
        ).fetchone()

        if not res:
            return

        template = res[0]

        # Replace placeholders
        template = template.replace("?", "'test'")

        try:
            explain = conn.execute(
                text(f"EXPLAIN {template}")
            ).fetchall()
        except Exception as e:
            st.warning(f"EXPLAIN failed: {e}")
            return

        explain_json = [dict(row._mapping) for row in explain]

        anti_patterns = []

        # Detect sequential scan
        for row in explain_json:
            if "Seq Scan" in str(row):
                anti_patterns.append("sequential_scan")

        conn.execute(text("""
            INSERT INTO execution_evidence
            (query_id, explain_output, anti_patterns)
            VALUES (:id, :exp, :anti)
        """), {
            "id": query_id,
            "exp": explain_json,
            "anti": anti_patterns
        })

        conn.commit()

        st.success("✅ Execution evidence generated")


# ─────────────────────────────────────────────
# Qdrant Ingestion
# ─────────────────────────────────────────────
def ingest_to_qdrant(chunks, env='dev'):

    if not chunks:
        st.warning("⚠️ No chunks to ingest into Qdrant")
        return

    client = get_qdrant_client(env)
    model = get_embedding_model()
    config = load_config(env)

    collection = config["qdrant"]["collection"]
    vector_size = config["vector_size"]

    create_qdrant_collection(client, collection, vector_size)

    points = []

    for chunk in chunks:
        if not chunk.get("content_text"):
            continue

        emb = model.encode(chunk["content_text"]).tolist()

        payload = {
            "chunkid": str(uuid.uuid4()),
            "content_text": chunk["content_text"],
            "project_name": config["project_name"],
            "environment": env
        }

        points.append(
            PointStruct(
                id=payload["chunkid"],
                vector=emb,
                payload=payload
            )
        )

    if not points:
        st.warning("⚠️ No valid points to upsert")
        return

    client.upsert(collection_name=collection, points=points)

    st.success(f"✅ {len(points)} vectors ingested to Qdrant")
```
