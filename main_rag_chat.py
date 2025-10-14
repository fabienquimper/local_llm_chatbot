import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import requests
from time import perf_counter
import argparse
import socket
import subprocess
import re

# ---------------------------
# Configuration
# ---------------------------
PDF_FOLDER = "./pdf"
CHROMA_DB = "vectordb"
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


def detect_wsl_windows_nameserver() -> str | None:
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if line.startswith("nameserver"):
                    return line.strip().split()[1]
    except Exception:
        return None
    return None


def get_default_gateway_ip() -> str | None:
    try:
        out = subprocess.check_output(["ip", "route"]).decode("utf-8", errors="ignore")
        for line in out.splitlines():
            if line.startswith("default "):
                parts = line.split()
                if "via" in parts:
                    idx = parts.index("via")
                    return parts[idx+1]
    except Exception:
        return None
    return None


def is_port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


# Pick LM Studio base URL with a fast TCP probe
_candidates = [LMSTUDIO_HOST]
if detect_wsl():
    gw = get_default_gateway_ip()
    if gw:
        _candidates.append(gw)
    ns = detect_wsl_windows_nameserver()
    if ns:
        _candidates.append(ns)
    _candidates.append("host.docker.internal")

_seen = set()
_hosts = [h for h in _candidates if not (h in _seen or _seen.add(h))]
LMSTUDIO_BASE_URL = None
for h in _hosts:
    if is_port_open(h, int(LMSTUDIO_PORT), timeout_s=1.5):
        LMSTUDIO_BASE_URL = f"{LMSTUDIO_SCHEME}://{h}:{LMSTUDIO_PORT}"
        break
if LMSTUDIO_BASE_URL is None:
    LMSTUDIO_BASE_URL = f"{LMSTUDIO_SCHEME}://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

print(f"🔗 LM Studio: {LMSTUDIO_BASE_URL}")

# ---------------------------
# 1) Load PDFs
# ---------------------------
def load_pdfs_from_folder(pdf_folder: str):
    docs = []
    for file in os.listdir(pdf_folder):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file)
            try:
                reader = PdfReader(file_path)
            except Exception:
                continue
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    docs.append({"source": os.path.basename(file_path), "page": page_num+1, "text": text})
    return docs


def load_pdfs_from_files(pdf_files: list[str]):
    docs = []
    for file_path in pdf_files:
        if not file_path.lower().endswith(".pdf"):
            continue
        if not os.path.isfile(file_path):
            continue
        try:
            reader = PdfReader(file_path)
        except Exception:
            continue
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                docs.append({"source": os.path.basename(file_path), "page": page_num+1, "text": text})
    return docs

# ---------------------------
# 2) Initialization + optional (re)indexing
# ---------------------------
parser = argparse.ArgumentParser(description="RAG Chat with ChromaDB and LM Studio")
parser.add_argument("-f", "--force-reindex", action="store_true", help="Delete and rebuild the vector database from PDFs")
parser.add_argument("--use-existing", action="store_true", help="Use existing vector DB without (re)indexing; error if missing")
parser.add_argument("--pdf-folder", type=str, default=PDF_FOLDER, help="Folder containing PDFs (default: ./pdf)")
parser.add_argument("--pdf-files", type=str, nargs="*", default=None, help="Explicit list of PDF files to index")
parser.add_argument("--db-path", type=str, default=CHROMA_DB, help="Destination path for ChromaDB persistent store")
parser.add_argument("--collection", type=str, default="docs", help="ChromaDB collection name (default: docs)")
parser.add_argument("--timeout", type=int, default=int(os.environ.get("LMSTUDIO_TIMEOUT", "240")), help="LM Studio API read timeout (s)")
parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("LMSTUDIO_MAX_TOKENS", "512")), help="Maximum number of tokens generated by LM Studio")
parser.add_argument("--stream", action="store_true", help="Enable streaming response (progressive display)")
args = parser.parse_args()

# Utilise un client persistant compatible avec les versions récentes de ChromaDB
chroma_client = chromadb.PersistentClient(path=args.db_path)

if args.force_reindex:
    try:
        chroma_client.delete_collection(name=args.collection)
        print("♻️ Existing collection removed (force-reindex)")
    except Exception:
        pass

try:
    collection = chroma_client.get_collection(name=args.collection)
    print("✅ Vector database found, no reindexing needed")
except Exception:
    if args.use_existing:
        raise SystemExit(f"❌ Collection '{args.collection}' not found in DB at '{args.db_path}', and --use-existing was set.")

    collection = chroma_client.create_collection(name=args.collection)
    print("⚡ Indexing in progress...")

    start_load = perf_counter()
    if args.pdf_files:
        pdfs = load_pdfs_from_files(args.pdf_files)
    else:
        pdfs = load_pdfs_from_folder(args.pdf_folder)
    end_load = perf_counter()
    print(f"✅ {len(pdfs)} pages loaded from PDFs in {end_load - start_load:.2f}s")

    start_add = perf_counter()
    for i, doc in enumerate(pdfs):
        collection.add(
            ids=[str(i)],
            documents=[doc["text"]],
            metadatas=[{"source": doc["source"], "page": doc["page"]}],
            embeddings=[embedding_model.encode(doc["text"]).tolist()]
        )
    end_add = perf_counter()

    print("✅ Indexing completed:", len(pdfs), f"passages in {end_add - start_add:.2f}s")

# ---------------------------
# 3) Retrieval + LM Studio
# ---------------------------
def retrieve_context(question, top_k=3):
    query_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    context = "\n\n".join(results["documents"][0])
    return context


def clean_lm_output(text: str) -> str:
    # Remove tags like <|...|> and cleanup
    text = re.sub(r"<\|[^>]*\|>", "", text)
    return text.strip()


def ask_lmstudio(question):
    context = retrieve_context(question)
    prompt = f"""You are an expert assistant.
Answer the question using the following documents:

{context}

Question: {question}
Answer:
"""

    payload = {
        "model": os.environ.get("LMSTUDIO_MODEL", "local-model"),
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "stream": bool(args.stream),
    }

    try:
        if args.stream:
            with requests.post(
                f"{LMSTUDIO_BASE_URL}/v1/completions",
                json=payload,
                timeout=(5, args.timeout),
                stream=True,
            ) as resp:
                resp.raise_for_status()
                collected = []
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data:" ):
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = requests.utils.json.loads(data)
                            delta = obj.get("choices", [{}])[0].get("text")
                            if delta:
                                print(delta, end="", flush=True)
                                collected.append(delta)
                        except Exception:
                            continue
                print()
                return clean_lm_output("".join(collected))
        else:
            response = requests.post(
                f"{LMSTUDIO_BASE_URL}/v1/completions",
                json=payload,
                timeout=(5, args.timeout),
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["text"]
            return clean_lm_output(text)
    except requests.exceptions.RequestException as exc:
        return (
            f"Unable to reach LM Studio at {LMSTUDIO_BASE_URL}. "
            f"Make sure the local server is running and listening on this port.\n"
            f"Details: {exc}"
        )

# ---------------------------
# 4) User loop
# ---------------------------
while True:
    question = input("\nAsk your question (or 'quit' to exit): ")
    if question.lower() in ["quit", "exit"]:
        break
    print("\n💡 Answer:", ask_lmstudio(question))


