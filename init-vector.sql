-- Enable pgvector on first boot so the `vector_memory.embedding` column and
-- the HNSW similarity index can be created by migrations / schema bootstrap.
CREATE EXTENSION IF NOT EXISTS vector;
