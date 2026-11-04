import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"
HANDBOOK_FILE = PROJECT_DIR / "company_policy_document.txt"


def load_env_file(path):
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()

        name, separator, value = line.partition("=")
        if not separator:
            continue

        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value


load_env_file(ENV_FILE)

MODEL = "qwen3:8b"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
SYSTEM_PROMPT_TEMPLATE = """You are a helpful company assistant. Answer the employee's question clearly.

COMPANY POLICY HANDBOOK:
{{policy}}"""


def build_system_prompt():
    handbook = HANDBOOK_FILE.read_text(encoding="utf-8")
    return SYSTEM_PROMPT_TEMPLATE.replace("{{policy}}", handbook)


def answer(question):
    request_body = json.dumps(
        {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": question},
            ],
        }
    ).encode("utf-8")
    request = Request(
        f"{OLLAMA_URL}/api/chat",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach Ollama at {OLLAMA_URL}: {error.reason}") from error

    return result.get("message", {}).get("content", "").strip() or (
        "The local model returned an empty response."
    )


def main():
    if not HANDBOOK_FILE.is_file():
        raise SystemExit(f"Company handbook not found: {HANDBOOK_FILE}")

    print(f"Company policy assistant ({MODEL}). Type 'exit' to quit.")
    while True:
        try:
            question = input("\nEmployee: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question or question.lower() in {"exit", "quit"}:
            break

        try:
            print(f"\nAssistant: {answer(question)}")
        except RuntimeError as error:
            print(f"\n{error}\nStart Ollama and ensure {MODEL} is installed.")


if __name__ == "__main__":
    main()