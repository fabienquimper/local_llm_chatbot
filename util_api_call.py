import os
import json
from urllib import request, error
import argparse
import socket
import subprocess


def fetch_models(base_url: str, timeout_s: int) -> dict:
    """Fetch the list of models from the LM Studio-compatible API."""
    req = request.Request(
        url=f"{base_url}/v1/models",
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "lmstudio-models-client/1.0",
        },
    )

    with request.urlopen(req, timeout=timeout_s) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        body = resp.read().decode(charset)
        return json.loads(body)


def complete_prompt(base_url: str, model: str, prompt: str, max_tokens: int = 256, timeout_s: int = 120) -> dict:
    """Send a completion request with the given prompt and return the JSON response."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    req = request.Request(
        url=f"{base_url}/v1/completions",
        method="POST",
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "lmstudio-completions-client/1.0",
        },
    )

    with request.urlopen(req, timeout=timeout_s) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        body = resp.read().decode(charset)
        return json.loads(body)


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


def pick_reachable_base_url(scheme: str, host: str, port: str, timeout_s: int) -> str:
    candidates: list[str] = []
    candidates.append(host)

    if detect_wsl():
        # Prioritize default gateway for WSL, then nameserver, then host.docker.internal
        gw = get_default_gateway_ip()
        if gw:
            candidates.append(gw)
        ns = detect_wsl_windows_nameserver()
        if ns:
            candidates.append(ns)
        candidates.append("host.docker.internal")

    # Deduplicate
    seen = set()
    unique_hosts: list[str] = []
    for h in candidates:
        if h and h not in seen:
            seen.add(h)
            unique_hosts.append(h)

    # Fast TCP probe instead of slow HTTP probe
    for h in unique_hosts:
        if is_port_open(h, int(port), timeout_s=min(2, max(1, timeout_s//60))):
            return f"{scheme}://{h}:{port}"

    return f"{scheme}://{host}:{port}"


def main() -> None:
    parser = argparse.ArgumentParser(description="LM Studio API helper")
    parser.add_argument("--prompt", type=str, help="Prompt to send to the model", default=None)
    parser.add_argument("--model", type=str, help="Model name (default: env LMSTUDIO_MODEL or 'local-model')", default=None)
    parser.add_argument("--max-tokens", type=int, help="Max number of generated tokens", default=256)
    parser.add_argument("--timeout", type=int, help="Request timeout in seconds", default=180)
    parser.add_argument("--host", type=str, help="LM Studio host (default env LMSTUDIO_HOST or 127.0.0.1)", default=None)
    parser.add_argument("--port", type=str, help="LM Studio port (default env LMSTUDIO_PORT or 1234)", default=None)
    parser.add_argument("--scheme", type=str, help="http/https scheme (default env LMSTUDIO_SCHEME or http)", default=None)
    args = parser.parse_args()

    host = args.host or os.environ.get("LMSTUDIO_HOST", "127.0.0.1")
    port = args.port or os.environ.get("LMSTUDIO_PORT", "1234")
    scheme = args.scheme or os.environ.get("LMSTUDIO_SCHEME", "http")

    base_url = pick_reachable_base_url(scheme, host, port, timeout_s=args.timeout)

    print(f"🔗 LM Studio: {base_url}")

    try:
        if args.prompt:
            model_name = args.model or os.environ.get("LMSTUDIO_MODEL", "local-model")
            resp = complete_prompt(base_url, model=model_name, prompt=args.prompt, max_tokens=args.max_tokens, timeout_s=args.timeout)
            try:
                text = resp.get("choices", [{}])[0].get("text")
                if text is not None:
                    print(text)
                else:
                    print(json.dumps(resp, ensure_ascii=False, indent=2))
            except Exception:
                print(json.dumps(resp, ensure_ascii=False, indent=2))
        else:
            models = fetch_models(base_url, timeout_s=args.timeout)
            print(json.dumps(models, ensure_ascii=False, indent=2))
    except socket.timeout:
        print(f"Timed out after {args.timeout}s. Increase --timeout or reduce --max-tokens.")
        raise SystemExit(1)
    except error.HTTPError as http_err:
        print(f"HTTP error {http_err.code}: {http_err.reason}")
        try:
            details = http_err.read().decode("utf-8", errors="ignore")
            if details:
                print(details)
        except Exception:
            pass
        raise SystemExit(1)
    except error.URLError as url_err:
        print(f"Connection error: {url_err.reason}. Is the server running on {base_url}?")
        raise SystemExit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()


