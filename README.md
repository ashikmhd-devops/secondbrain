<div align="center">

# 🧠 SecondBrain

**Your personal memory system — local, private, and AI-powered.**

Store anything. Recall everything. No cloud. No subscriptions. No data leaving your machine.

---

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?style=flat-square)](https://ollama.com)
[![SQLite](https://img.shields.io/badge/SQLite-FTS5%20%2B%20embeddings-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## What is it?

SecondBrain is a self-hosted, AI-augmented memory vault. Dump raw text — passwords, account numbers, API keys, notes, ideas, contacts, events — and retrieve it later in plain English. The LLM runs entirely on your machine via Ollama.

---

## Features

### 🔍 Two-Mode Search
| Mode | How it works |
|---|---|
| **Semantic recall** | Embeds your question and finds memories by meaning, not keywords. Passes top matches to the LLM for a synthesised answer. |
| **Full-text search** | SQLite FTS5 index for exact keyword and phrase lookups. |

### 🤖 Auto-Enrichment on Save
Every memory is automatically analysed by the local LLM and enriched with:
- **Title** — short, descriptive (max 80 chars)
- **Category** — `Person` · `Finance` · `Credentials` · `Event` · `Idea` · `Task` · `Health` · `Travel` · `Reference` · `General`
- **Tags** — relevant, searchable keywords
- **Data types** — detected value types (`account_number`, `api_key`, `password`, `email`, `url`, `date`, `currency`, and 20+ more)
- **Extracted entities** — names, amounts, dates, phones, emails, URLs, account numbers, card numbers, IFSC codes, UPI IDs, PAN numbers, API keys, passwords, addresses, and more
- **Sensitivity flag** — automatically marked 🔒 if it contains credentials, keys, or financial data

### 📅 Upcoming Events Widget
Memories categorised as `Event` with future dates surface automatically on the home screen, sorted by days remaining — color-coded by urgency (today → 2 days → 7 days → beyond).

### 🔒 Sensitive Memory Protection
Sensitive memories are blurred in the browse view. Click to reveal for 10 seconds, then they re-blur automatically.

### 🗂️ Browse & Filter
- Filter by **category**, **tag**, or **sensitive-only**
- Paginated grid view (12 per page, up to 500)
- Inline **edit** with automatic re-enrichment and re-embedding on save
- One-click **delete**

### 🔐 Optional Authentication
Set `APP_PASSWORD` in `.env` to enable password-based login. Generates secure 32-byte session tokens valid for 7 days. Leave it blank for open localhost access.

### 🐳 Docker Ready
Single `docker compose up` — no Python setup required.

### 📱 Mobile Friendly
Access from any device on the same WiFi network.

---

## Prerequisites

```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

---

## Quickstart

### Local
```bash
pip install -r requirements.txt
cp .env.example .env   # edit as needed
python app.py
```

### Docker
```bash
docker compose up
```

Open **http://localhost:8765**

> **Docker note:** Ollama must be running on the host — the container reaches it via `host.docker.internal:11434`.

---

## Access from Phone or Tablet

Find your machine's local IP:

```bash
# macOS
ipconfig getifaddr en0

# Linux
hostname -I | awk '{print $1}'
```

Then open `http://<your-ip>:8765` on any device on the same network.

---

## Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for semantic search |
| `CHAT_MODEL` | `llama3.2:latest` | Chat model for enrichment and recall |
| `DB_PATH` | `secondbrain.db` | SQLite database path |
| `PORT` | `8765` | Server port |
| `APP_PASSWORD` | _(empty)_ | Set to enable login protection |

---

## Backup

Your entire database is a single file:

```bash
cp secondbrain.db secondbrain.backup.db
```

Copy it anywhere to restore.

---

## API Reference

All `/api/*` routes require a `Bearer` token when `APP_PASSWORD` is set.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/memory` | Store and auto-enrich a new memory |
| `GET` | `/api/memory` | List memories — filter by `category`, `tag`, `sensitive` |
| `GET` | `/api/memory/{id}` | Get a single memory with extracted entities |
| `PUT` | `/api/memory/{id}` | Update and re-enrich a memory |
| `DELETE` | `/api/memory/{id}` | Delete a memory |
| `POST` | `/api/recall` | Semantic search + LLM-generated answer |
| `POST` | `/api/search` | Full-text keyword search |
| `GET` | `/api/events/upcoming` | Upcoming events sorted by days remaining |
| `GET` | `/api/stats` | Total count, sensitive count, category breakdown, recent items |
| `GET` | `/api/tags` | All tags sorted by frequency |
| `GET` | `/api/models` | Ollama connectivity and model availability check |
| `GET` | `/health` | Health check |

### Auth endpoints (public)

| Method | Path | Description |
|---|---|---|
| `GET` | `/auth/status` | Whether auth is enabled |
| `POST` | `/auth/login` | Exchange password for session token |
| `POST` | `/auth/logout` | Revoke current token |

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Database | SQLite with FTS5 virtual table and binary embedding blobs |
| AI / Embeddings | Ollama (`nomic-embed-text` + `llama3.2`) |
| Similarity | NumPy cosine similarity |
| Frontend | Vanilla JS, no framework, no build step |
| Auth | In-memory token store, `secrets`-based, constant-time comparison |
