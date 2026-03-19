# SecondBrain 🧠

A local-first personal memory system powered by Ollama. Store anything — notes, passwords, account numbers, API keys, ideas — and recall it in plain English.

All data stays on your machine. No cloud. No subscriptions.

---

## Prerequisites

```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

---

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8765

---

## Run via Docker

```bash
docker compose up
```

Open http://localhost:8765

> Ollama must be running on the host — Docker reaches it via `host.docker.internal:11434`.

---

## Access from phone or tablet (same WiFi)

Find your machine's local IP:

```bash
# macOS
ipconfig getifaddr en0

# Linux
hostname -I | awk '{print $1}'
```

Then open `http://<your-machine-ip>:8765` on any device on the same network.

---

## Backup

Your entire database is a single file:

```bash
cp secondbrain.db secondbrain.backup.db
```

That's it. Copy it anywhere to restore.

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHAT_MODEL` | `llama3.2:latest` | Chat/generation model |
| `DB_PATH` | `secondbrain.db` | SQLite database path |
| `PORT` | `8765` | Server port |

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/memory` | Store a new memory |
| `GET` | `/api/memory` | List memories (filter by category, tag, sensitive) |
| `GET` | `/api/memory/{id}` | Get a single memory with entities |
| `PUT` | `/api/memory/{id}` | Update and re-enrich a memory |
| `DELETE` | `/api/memory/{id}` | Delete a memory |
| `POST` | `/api/recall` | Semantic search + LLM answer |
| `POST` | `/api/search` | Full-text keyword search |
| `GET` | `/api/stats` | Counts, categories, recent items |
| `GET` | `/api/tags` | All tags sorted by frequency |
| `GET` | `/health` | Health check |
