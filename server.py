import os
import shutil
import uuid
import json
from typing import List, Optional, Dict, Any
import re
import threading
import time
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chromadb
import requests
import socket
import subprocess
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Basic paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")  # legacy/global uploads (kept for compatibility)
STATIC_DIR = os.path.join(BASE_DIR, "static")
BASES_DIR = os.path.join(BASE_DIR, "bases")
BASES_FILE = os.path.join(BASE_DIR, "bases.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(BASES_DIR, exist_ok=True)
if not os.path.isfile(BASES_FILE):
    with open(BASES_FILE, "w", encoding="utf-8") as f:
        json.dump({"bases": {}}, f)


class IndexRequest(BaseModel):
    db_path: str
    collection: str = "docs"
    pdf_folder: Optional[str] = None
    pdf_files: Optional[List[str]] = None
    force_reindex: bool = False


class ChatRequest(BaseModel):
    db_path: str
    collection: str = "docs"
    question: str
    top_k: int = 3
    model: Optional[str] = None


class ConfigRequest(BaseModel):
    db_path: str
    collection: str


app = FastAPI(title="Local RAG Chat Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI under /ui and keep /api routes unaffected
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="static")


# Global model singletons
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
AVG_EMB_TIME_PER_PAGE_S = float(os.environ.get("AVG_EMB_TIME_PER_PAGE_S", "0.05"))


# LM Studio connection helpers
LMSTUDIO_HOST = os.environ.get("LMSTUDIO_HOST", "127.0.0.1")
LMSTUDIO_PORT = os.environ.get("LMSTUDIO_PORT", "1234")
LMSTUDIO_SCHEME = os.environ.get("LMSTUDIO_SCHEME", "http")


def detect_wsl() -> bool:
    try:
        with open("/proc/sys/kernel/osrelease", "r") as f:
            osrelease = f.read().lower()
        return "microsoft" in osrelease or "wsl" in osrelease
    except Exception:
        return False


def detect_wsl_windows_nameserver() -> Optional[str]:
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if line.startswith("nameserver"):
                    return line.strip().split()[1]
    except Exception:
        return None
    return None


def get_default_gateway_ip() -> Optional[str]:
    try:
        out = subprocess.check_output(["ip", "route"]).decode("utf-8", errors="ignore")
        for line in out.splitlines():
            if line.startswith("default "):
                parts = line.split()
                if "via" in parts:
                    idx = parts.index("via")
                    return parts[idx + 1]
    except Exception:
        return None
    return None


def is_port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


def pick_reachable_base_url() -> str:
    candidates = [LMSTUDIO_HOST]
    if detect_wsl():
        gw = get_default_gateway_ip()
        if gw:
            candidates.append(gw)
        ns = detect_wsl_windows_nameserver()
        if ns:
            candidates.append(ns)
        candidates.append("host.docker.internal")
    seen = set()
    unique_hosts = [h for h in candidates if not (h in seen or seen.add(h))]
    for h in unique_hosts:
        if is_port_open(h, int(LMSTUDIO_PORT), timeout_s=1.5):
            return f"{LMSTUDIO_SCHEME}://{h}:{LMSTUDIO_PORT}"
    return f"{LMSTUDIO_SCHEME}://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}"


# ===== Bases registry helpers =====
def load_bases() -> Dict[str, Any]:
    try:
        with open(BASES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict) or "bases" not in data:
                return {"bases": {}}
            return data
    except Exception:
        return {"bases": {}}


def save_bases(data: Dict[str, Any]) -> None:
    with open(BASES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


_VALID_COLLECTION_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{1,510})[A-Za-z0-9]$")


def is_valid_collection_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    return bool(_VALID_COLLECTION_RE.match(name))


def sanitize_collection_name(display_name: str) -> str:
    # Replace disallowed chars with '-'
    name = re.sub(r"[^A-Za-z0-9._-]", "-", display_name)
    # Trim leading/trailing non-alnum
    name = re.sub(r"^[^A-Za-z0-9]+", "", name)
    name = re.sub(r"[^A-Za-z0-9]+$", "", name)
    # Ensure minimum length 3
    if len(name) < 3:
        name = (name + "base")[:3]
    # Truncate to 64 chars to be safe
    name = name[:64]
    # If still invalid (edge cases), fallback to deterministic slug
    if not is_valid_collection_name(name):
        base = re.sub(r"[^A-Za-z0-9]", "", display_name) or "base"
        base = base[:30] or "base"
        name = f"{base}-col"
        name = name[:64]
        if len(name) < 3:
            name = (name + "col")[:3]
    # Final guard
    if not is_valid_collection_name(name):
        name = "base-col"
    return name


def list_collections(db_path: str) -> list[str]:
    client = chromadb.PersistentClient(path=db_path)
    cols = client.list_collections()
    return [c.name for c in cols]


def load_docs_from_folder(pdf_folder: str) -> list[dict]:
    docs: list[dict] = []
    if not pdf_folder or not os.path.isdir(pdf_folder):
        return docs
    for name in os.listdir(pdf_folder):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_folder, name)
        try:
            reader = PdfReader(path)
        except Exception:
            continue
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                docs.append({"source": os.path.basename(path), "page": page_num + 1, "text": text})
    return docs


def load_docs_from_files(pdf_files: Optional[List[str]]) -> list[dict]:
    docs: list[dict] = []
    if not pdf_files:
        return docs
    for path in pdf_files:
        if not path or not path.lower().endswith(".pdf"):
            continue
        if not os.path.isfile(path):
            continue
        try:
            reader = PdfReader(path)
        except Exception:
            continue
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                docs.append({"source": os.path.basename(path), "page": page_num + 1, "text": text})
    return docs


@app.get("/api/collections")
def api_collections(db_path: str):
    return {"db_path": db_path, "collections": list_collections(db_path)}


@app.post("/api/config")
def api_config_set(cfg: ConfigRequest):
    # Stateless: just echo back; client stores selection
    return {"ok": True, "db_path": cfg.db_path, "collection": cfg.collection}


@app.post("/api/upload")
async def api_upload(files: List[UploadFile] = File(...)):
    saved: list[str] = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        uid = uuid.uuid4().hex
        dest = os.path.join(UPLOAD_DIR, f"{uid}_{os.path.basename(f.filename)}")
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
    return {"saved": saved}


# ===== Base-centric API =====
@app.get("/api/bases")
def bases_list():
    data = load_bases()
    out = []
    for name, meta in data.get("bases", {}).items():
        out.append({
            "name": name,
            "pdf_files": meta.get("pdf_files", []),
            "indexed": bool(meta.get("indexed", False)),
            "db_path": meta.get("db_path"),
            "collection": meta.get("collection"),
            "indexing": meta.get("indexing"),
        })
    return {"bases": out}


class CreateBaseRequest(BaseModel):
    name: str
    db_path: Optional[str] = None
    collection: Optional[str] = None


@app.post("/api/bases")
def bases_create(req: CreateBaseRequest):
    name = req.name.strip()
    if not name:
        return {"ok": False, "error": "Base name required"}
    data = load_bases()
    if name in data.get("bases", {}):
        return {"ok": False, "error": "Base already exists"}
    collection = req.collection or sanitize_collection_name(name)
    base_slug = collection  # use sanitized collection as folder slug
    base_dir = os.path.join(BASES_DIR, base_slug)
    pdf_dir = os.path.join(base_dir, "pdfs")
    db_dir = os.path.join(base_dir, "vectordb")
    db_path = req.db_path or db_dir
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    data["bases"][name] = {
        "pdf_files": [],
        "indexed": False,
        "db_path": db_path,
        "collection": collection,
        "base_dir": base_dir,
        "pdf_dir": pdf_dir,
    }
    save_bases(data)
    return {"ok": True, "base": {"name": name, **data["bases"][name]}}


@app.get("/api/bases/{name}")
def bases_get(name: str):
    data = load_bases()
    meta = data.get("bases", {}).get(name)
    if not meta:
        return {"ok": False, "error": "Not found"}
    return {"ok": True, "base": {"name": name, **meta}}


@app.delete("/api/bases/{name}")
def bases_delete(name: str):
    data = load_bases()
    if name in data.get("bases", {}):
        del data["bases"][name]
        save_bases(data)
        return {"ok": True}
    return {"ok": False, "error": "Not found"}


@app.post("/api/bases/{name}/upload")
async def bases_upload(name: str, files: List[UploadFile] = File(...)):
    data = load_bases()
    meta = data.get("bases", {}).get(name)
    if not meta:
        return {"ok": False, "error": "Base not found"}
    # Ensure base-specific pdf directory exists
    pdf_dir = meta.get("pdf_dir")
    if not pdf_dir:
        base_slug = meta.get("collection") or sanitize_collection_name(name)
        base_dir = os.path.join(BASES_DIR, base_slug)
        pdf_dir = os.path.join(base_dir, "pdfs")
        os.makedirs(pdf_dir, exist_ok=True)
        meta["base_dir"] = base_dir
        meta["pdf_dir"] = pdf_dir
    saved: list[str] = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        uid = uuid.uuid4().hex
        dest = os.path.join(pdf_dir, f"{uid}_{os.path.basename(f.filename)}")
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
        meta.setdefault("pdf_files", []).append(dest)
        meta["indexed"] = False
    save_bases(data)
    return {"ok": True, "saved": saved}


@app.post("/api/bases/{name}/index")
def bases_index(name: str):
    data = load_bases()
    meta = data.get("bases", {}).get(name)
    if not meta:
        return {"ok": False, "error": "Base not found"}
    # Auto-sanitize collection if needed
    collection_name = meta.get("collection") or sanitize_collection_name(name)
    if not is_valid_collection_name(collection_name):
        collection_name = sanitize_collection_name(collection_name)
        meta["collection"] = collection_name
    # Keep db_path stable; if missing, derive inside base directory
    base_dir = meta.get("base_dir")
    if not base_dir:
        base_dir = os.path.join(BASES_DIR, collection_name)
        meta["base_dir"] = base_dir
    db_dir = os.path.join(base_dir, "vectordb")
    os.makedirs(db_dir, exist_ok=True)
    db_path = meta.get("db_path") or db_dir
    meta["db_path"] = db_path
    pdf_files = meta.get("pdf_files", [])

    client = chromadb.PersistentClient(path=db_path)
    # Clean existing collection for this base
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)

    # If already indexing, prevent duplicate
    if meta.get("indexing", {}).get("status") == "running":
        return {"ok": False, "error": "Indexing already in progress"}

    # Pre-count pages for estimation
    total_pages = 0
    for path in pdf_files:
        try:
            reader = PdfReader(path)
            total_pages += len(reader.pages)
        except Exception:
            continue

    meta["indexing"] = {
        "status": "running",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "total_pages": total_pages,
        "processed_pages": 0,
        "eta_seconds": int(total_pages * AVG_EMB_TIME_PER_PAGE_S),
    }
    save_bases(data)

    def worker():
        nonlocal meta
        client_local = chromadb.PersistentClient(path=db_path)
        try:
            try:
                client_local.delete_collection(name=collection_name)
            except Exception:
                pass
            collection_local = client_local.create_collection(name=collection_name)

            processed = 0
            page_id = 0
            for file_path in pdf_files:
                try:
                    reader = PdfReader(file_path)
                except Exception:
                    continue
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    doc = {"source": os.path.basename(file_path), "page": page_num + 1, "text": text}
                    collection_local.add(
                        ids=[str(page_id)],
                        documents=[doc["text"]],
                        metadatas=[{"source": doc["source"], "page": doc["page"]}],
                        embeddings=[embedding_model.encode(doc["text"]).tolist()],
                    )
                    page_id += 1
                    processed += 1
                    # Update progress occasionally
                    if processed % 5 == 0 or processed == total_pages:
                        data_local = load_bases()
                        m = data_local.get("bases", {}).get(name, {})
                        idx = m.get("indexing", {})
                        idx["processed_pages"] = processed
                        # Rough remaining time
                        remaining = max(0, (total_pages - processed) * AVG_EMB_TIME_PER_PAGE_S)
                        idx["eta_seconds"] = int(remaining)
                        m["indexing"] = idx
                        save_bases({"bases": {**data_local.get("bases", {}), name: m}})

            # Finished
            data_done = load_bases()
            m = data_done.get("bases", {}).get(name, {})
            m["indexed"] = True
            m["indexing"] = {"status": "done", "total_pages": total_pages, "processed_pages": processed, "eta_seconds": 0}
            save_bases({"bases": {**data_done.get("bases", {}), name: m}})
        except Exception:
            data_err = load_bases()
            m = data_err.get("bases", {}).get(name, {})
            m["indexing"] = {"status": "error"}
            save_bases({"bases": {**data_err.get("bases", {}), name: m}})

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "started": True, "total_pages": total_pages}


class BaseChatRequest(BaseModel):
    question: str
    top_k: int = 3
    model: Optional[str] = None


@app.post("/api/bases/{name}/chat")
def bases_chat(name: str, req: BaseChatRequest):
    data = load_bases()
    meta = data.get("bases", {}).get(name)
    if not meta:
        return {"ok": False, "error": "Base not found"}
    if not meta.get("indexed"):
        return {"ok": False, "error": "Base is not indexed"}
    db_path = meta.get("db_path")
    collection_name = meta.get("collection", name)

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)

    query_embedding = embedding_model.encode(req.question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=req.top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context = "\n\n".join(documents)
    base_url = pick_reachable_base_url()
    model_name = req.model or os.environ.get("LMSTUDIO_MODEL", "local-model")
    prompt = (
        "You are an expert assistant. Answer the question using the following documents.\n\n"
        f"{context}\n\nQuestion: {req.question}\nAnswer:"
    )
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": int(os.environ.get("LMSTUDIO_MAX_TOKENS", "512")),
        "stream": False,
    }
    try:
        resp = requests.post(f"{base_url}/v1/completions", json=payload, timeout=(5, int(os.environ.get("LMSTUDIO_TIMEOUT", "240"))))
        resp.raise_for_status()
        text = resp.json().get("choices", [{}])[0].get("text", "")
        answer = text.strip()
    except Exception as exc:
        answer = f"[LM Studio error] {exc}"
    return {"ok": True, "answer": answer, "context": documents, "metadatas": metadatas}


@app.get("/api/bases/{name}/status")
def bases_status(name: str):
    data = load_bases()
    meta = data.get("bases", {}).get(name)
    if not meta:
        return {"ok": False, "error": "Base not found"}
    return {"ok": True, "indexed": bool(meta.get("indexed")), "indexing": meta.get("indexing")}


@app.get("/api/bases/{name}/pdfs")
def bases_pdfs(name: str):
    data = load_bases()
    meta = data.get("bases", {}).get(name)
    if not meta:
        return {"ok": False, "error": "Base not found"}
    pdf_files = meta.get("pdf_files", [])
    items = []
    for p in pdf_files:
        base = os.path.basename(p)
        items.append({"file": base, "url": f"/api/bases/{name}/pdf/{base}"})
    return {"ok": True, "pdfs": items}


@app.get("/api/bases/{name}/pdf/{filename}")
def bases_pdf_download(name: str, filename: str):
    data = load_bases()
    meta = data.get("bases", {}).get(name)
    if not meta:
        return {"ok": False, "error": "Base not found"}
    pdf_dir = meta.get("pdf_dir")
    if not pdf_dir:
        return {"ok": False, "error": "No pdf_dir for base"}
    # Sanitize filename and serve
    safe_name = os.path.basename(filename)
    path = os.path.join(pdf_dir, safe_name)
    if not os.path.isfile(path):
        return {"ok": False, "error": "File not found"}
    return FileResponse(path, media_type="application/pdf", filename=safe_name)


@app.post("/api/index")
def api_index(req: IndexRequest):
    client = chromadb.PersistentClient(path=req.db_path)

    if req.force_reindex:
        try:
            client.delete_collection(name=req.collection)
        except Exception:
            pass

    try:
        collection = client.get_collection(name=req.collection)
        did_exist = True
    except Exception:
        collection = client.create_collection(name=req.collection)
        did_exist = False

    # If collection existed and not force_reindex, we still allow adding docs
    docs = []
    if req.pdf_files:
        docs.extend(load_docs_from_files(req.pdf_files))
    if req.pdf_folder:
        docs.extend(load_docs_from_folder(req.pdf_folder))

    start_id = 0
    if did_exist and not req.force_reindex:
        # Attempt to continue IDs after existing count; if count not available, just append from 0
        try:
            existing = collection.count()
            start_id = int(existing)
        except Exception:
            start_id = 0

    for i, doc in enumerate(docs):
        collection.add(
            ids=[str(start_id + i)],
            documents=[doc["text"]],
            metadatas=[{"source": doc["source"], "page": doc["page"]}],
            embeddings=[embedding_model.encode(doc["text"]).tolist()],
        )

    return {"ok": True, "added": len(docs)}


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    client = chromadb.PersistentClient(path=req.db_path)
    collection = client.get_collection(name=req.collection)

    query_embedding = embedding_model.encode(req.question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=req.top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context = "\n\n".join(documents)

    # Build prompt and query LM Studio
    base_url = pick_reachable_base_url()
    model_name = req.model or os.environ.get("LMSTUDIO_MODEL", "local-model")
    prompt = (
        "You are an expert assistant. Answer the question using the following documents.\n\n"
        f"{context}\n\nQuestion: {req.question}\nAnswer:"
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": int(os.environ.get("LMSTUDIO_MAX_TOKENS", "512")),
        "stream": False,
    }

    try:
        resp = requests.post(f"{base_url}/v1/completions", json=payload, timeout=(5, int(os.environ.get("LMSTUDIO_TIMEOUT", "240"))))
        resp.raise_for_status()
        text = resp.json().get("choices", [{}])[0].get("text", "")
        answer = text.strip()
    except Exception as exc:
        answer = f"[LM Studio error] {exc}"

    return {"answer": answer, "context": documents, "metadatas": metadatas}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root_index():
    # Serve UI index at root
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/favicon.ico")
def favicon():
    icon_path = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.isfile(icon_path):
        return FileResponse(icon_path)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


