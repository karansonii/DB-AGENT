```python
import os
import yaml
from functools import lru_cache
import streamlit as st
from sqlalchemy import create_engine
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv

# ────────────────────────────────────────────────
# Load .env file (API keys, etc.)
# This must be called early – preferably at module level
# ────────────────────────────────────────────────
load_dotenv()  # Automatically loads .env from project root


@lru_cache()
def load_config(env: str = "dev"):
    """
    Load and merge base.yaml + env-specific yaml.
    Supports environment variable overrides.
    """
    base_path = "config/base.yaml"
    env_path = f"config/{env}.yaml"

    config = {}

    # Load base with UTF-8 to avoid encoding issues on Windows
    with open(base_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    # Load env-specific config if exists
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            env_config = yaml.safe_load(f) or {}
            # Deep merge helper
            def merge(d1, d2):
                for k, v in d2.items():
                    if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
                        d1[k] = merge(d1[k], v)
                    else:
                        d1[k] = v
                return d1
            config = merge(config, env_config)

    # Environment variable overrides (highest priority)
    env_overrides = {
        'MYSQL_HOST': ('mysql', 'host'),
        'MYSQL_PORT': ('mysql', 'port'),
        'MYSQL_USER': ('mysql', 'user'),
        'MYSQL_PASSWORD': ('mysql', 'password'),
        'MYSQL_DB': ('mysql', 'db'),
        'POSTGRES_HOST': ('postgres', 'host'),
        'POSTGRES_PORT': ('postgres', 'port'),
        'POSTGRES_USER': ('postgres', 'user'),
        'POSTGRES_PASSWORD': ('postgres', 'password'),
        'POSTGRES_DB': ('postgres', 'db'),
        'QDRANT_HOST': ('qdrant', 'host'),
        'QDRANT_PORT': ('qdrant', 'port'),
    }

    for env_key, (section, subkey) in env_overrides.items():
        value = os.getenv(env_key)
        if value is not None:
            config.setdefault(section, {})[subkey] = value

    # Replace {{env}} placeholders
    for section in ["mysql", "postgres", "qdrant"]:
        if section in config:
            for k, v in list(config[section].items()):  # list() to avoid runtime modification error
                if isinstance(v, str):
                    config[section][k] = v.replace("{{env}}", env)

    config["current_env"] = env

    return config


@st.cache_resource
@st.cache_resource
def get_mysql_engine(_env: str):
    """
    Redirect MySQL calls to PostgreSQL since we only use Postgres.
    """
    return get_postgres_engine(_env)



@st.cache_resource
def get_postgres_engine(_env: str):
    cfg = load_config(_env)["postgres"]
    url = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['db']}"
    return create_engine(url)


@st.cache_resource
def get_qdrant_client(env='dev'):
    cfg = load_config(env)["qdrant"]
    return QdrantClient(
        host=cfg["host"],
        port=cfg["port"],
        check_compatibility=False
    )



@st.cache_resource
def get_embedding_model():
    model_name = load_config()["embedding_model"]
    return SentenceTransformer(model_name)


@st.cache_resource
def get_llm():
    """
    Get Gemini LLM instance using API key from .env file.
    No more st.secrets dependency.
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found!\n"
            "Please add it to your .env file in the project root like this:\n"
            "GOOGLE_API_KEY=AIzaSyYourRealKeyHereXXXXXXXXXXXXXXXXXXXXX\n"
            "Then restart Streamlit."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # or "gemini-1.5-flash" for faster/cheaper
        google_api_key=api_key,
        temperature=0.4
    )



def create_qdrant_collection(client, collection_name, vector_size):

    # ✅ New API
    exists = client.collection_exists(collection_name)

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")
```
