import streamlit as st
import pandas as pd
import json

from src.utils import (
    load_config,
    get_postgres_engine,
    get_qdrant_client,
)

from src.ingestion import (
    create_pg_tables,
    ingest_metadata_from_postgres,
    generate_evidence,
    ingest_to_qdrant,
)

from sqlalchemy import text
from src.agent import run_question_query, run_sql_query


# ────────────────────────────────────────────────
# Streamlit Setup
# ────────────────────────────────────────────────
st.set_page_config(page_title="DB Agent Platform for eGP", layout="wide")
st.title("DB Agent Platform for eGP")

# Environment selector
available_envs = ["dev", "qa", "uat", "prod"]
current_env = st.sidebar.selectbox("Environment", available_envs, index=0)
config = load_config(current_env)


# ────────────────────────────────────────────────
# 🔄 AUTO INGEST ON APP START
# ────────────────────────────────────────────────
def auto_ingest_pipeline(env):
    engine = get_postgres_engine(env)

    # 1️⃣ Ensure metadata tables exist
    create_pg_tables(env)

    # 2️⃣ Clear old metadata (DEV safe reset)
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM tables_metadata"))
        conn.execute(text("DELETE FROM execution_evidence"))
        conn.commit()

    # 3️⃣ Re-ingest schema metadata
    ingest_metadata_from_postgres(env)

    # 4️⃣ Generate evidence for existing queries
    df = pd.read_sql("SELECT id FROM queries", engine)
    for qid in df["id"].tolist():
        generate_evidence(qid, env)

    # 5️⃣ Read everything for embedding
    metadata_df = pd.read_sql("SELECT * FROM tables_metadata", engine)
    evidence_df = pd.read_sql("SELECT * FROM execution_evidence", engine)

    chunks = []

    # Metadata chunks
    for _, row in metadata_df.iterrows():
        chunks.append({
            "content_text": f"""
Table: {row.table_schema}.{row.table_name}
Column: {row.column_name}
Data Type: {row.data_type}
Nullable: {row.nullable}
Default: {row.default_value}
"""
        })

    # Evidence chunks (fixed bug here)
    for _, row in evidence_df.iterrows():
        chunks.append({
            "content_text": json.dumps(row.to_dict(), default=str)
        })

    # 6️⃣ Reset Qdrant collection
    client = get_qdrant_client(env)
    try:
        client.delete_collection(config["qdrant"]["collection"])
    except Exception:
        pass

    ingest_to_qdrant(chunks, env)


# Run only once per session
if "auto_ingested" not in st.session_state:
    st.info("🔄 Auto-ingesting DB schema & knowledge...")
    auto_ingest_pipeline(current_env)
    st.success("✅ Auto ingestion complete!")
    st.session_state.auto_ingested = True


# ────────────────────────────────────────────────
# 🤖 AGENT UI
# ────────────────────────────────────────────────

st.subheader("Ask Question (Natural Language Mode)")
question_input = st.text_input(
    "Example: What tables are available?"
)

if st.button("Run Question"):
    if question_input.strip():
        response = run_question_query(question_input, current_env)
        st.write("Response:", response)
    else:
        st.warning("Please enter a question.")


st.markdown("---")

st.subheader("Run SQL Query (Strict Mode)")
sql_input = st.text_area(
    "Enter SQL query (strict validation — no auto-correction)",
    height=150
)

if st.button("Run SQL"):
    if sql_input.strip():
        response = run_sql_query(sql_input, current_env)
        st.write("Response:", response)
    else:
        st.warning("Please enter a SQL query.")


# ────────────────────────────────────────────────
# 🔍 READ-ONLY DATA VIEW
# ────────────────────────────────────────────────
st.subheader("View PostgreSQL Metadata Tables")

engine = get_postgres_engine(current_env)


table_to_view = st.selectbox(
    "Select Table",
    ["tables_metadata", "queries", "execution_evidence"]
)

df = pd.read_sql(f"SELECT * FROM {table_to_view} LIMIT 20", engine)
st.dataframe(df)
