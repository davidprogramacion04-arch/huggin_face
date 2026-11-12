"""
Agente GAIA 

Uso:
  python app.py                         # chat de prueba en el navegador (http://127.0.0.1:7860)
  python app.py --gaia --limit 2        # prueba con 2 preguntas del reto
  python app.py --gaia                  # corre todas (guarda/reanuda en answers.json)
  python app.py --gaia --submit         # corre lo que falte y envía al scoring
  python app.py --submit-only           # envía lo ya guardado en answers.json
  python app.py --gaia --task-id ID --force   # repetir una pregunta concreta
"""
import argparse
import json
import os
import re
import threading
import time

import requests
from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    FinalAnswerTool,
    GradioUI,
    LiteLLMModel,
    VisitWebpageTool,
    tool,
)

import litellm
litellm.drop_params = True  

load_dotenv()  

API = "https://agents-course-unit4-scoring.hf.space"
ANSWERS_FILE = "answers.json"
DOWNLOADS = "downloads"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp")

# Reglas de formato de GAIA: la evaluación es por coincidencia EXACTA.
GAIA_RULES = """
You are a general AI assistant answering GAIA benchmark questions.
- Use wikipedia_page for Wikipedia questions, web_search + visit_webpage for other facts, read_file for attached files, and Python for any calculation.
- Verify important facts in at least one source before answering.
- NEVER call final_answer in the same code block where you first print or inspect data. Print the data, read the output in the next step, and only then call final_answer with an answer taken from that output (not from memory).
- If a page says 'not found', try another title or web_search; do not answer from memory.
- If you cannot find the answer after several attempts, give your best guess instead of leaving it empty.
- Call final_answer with ONLY the answer, as a string. No explanation, no sentence, no "FINAL ANSWER:".
- Numbers: digits only, no thousands separators, no units or % or $ unless the question asks for them.
- Text: as few words as possible, no articles, no abbreviations (write city names in full).
- Lists: comma-separated, applying the rules above to each element, in the order requested.
""".strip()


# ----------------------------------------------------------------------------
# HERRAMIENTAS DEL MODELO
#   1) web_search               (DuckDuckGo con reintentos + respaldo Wikipedia)
#   1b) wikipedia_page         (texto de un artículo de Wikipedia vía API)
#   2) visit_webpage       (leer páginas/Wikipedia en markdown, con tablas)
#   3) read_file                 (leer Excel/CSV/PDF/texto adjuntos)
#   4) final_answer         (devolver la respuesta)
# ----------------------------------------------------------------------------
@tool
def read_file(path: str) -> str:
    """Reads a local file (attached to the task) and returns its content as text.
    Supports .xlsx/.xls, .csv, .pdf and plain-text files (.txt, .md, .json, .py, .html, ...).
    Args:
        path: local path of the file, exactly as given in the task.
    """
    if not os.path.exists(path):
        return f"File not found: {path}"
    ext = os.path.splitext(path)[1].lower()
    limit = 20000
    try:
        if ext in (".xlsx", ".xlsm", ".xls"):
            import pandas as pd
            sheets = pd.read_excel(path, sheet_name=None, header=None)
            text = "\n\n".join(f"## Sheet: {n}\n{df.to_string()}" for n, df in sheets.items())
        elif ext in (".csv", ".tsv"):
            import pandas as pd
            text = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",").to_string()
        elif ext == ".pdf":
            from pypdf import PdfReader
            text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        elif ext in (".mp3", ".wav", ".m4a", ".ogg", ".mp4"):
            return "Audio/video files are not supported by this tool."
        elif ext in IMAGE_EXT:
            return "Images are attached directly to the task: look at the image instead of reading it."
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
    except Exception as e:  
        return f"Error reading {path}: {e}"
    if len(text) > limit:
        text = text[:limit] + f"\n...[truncated, {len(text)} chars total]"
    return text



@tool
def web_search(query: str) -> str:
    """Searches the web (DuckDuckGo, with automatic retries) and returns the top results as text.
    If DuckDuckGo fails, falls back to a Wikipedia search.
    Args:
        query: the search query.
    """
    last_err = None
    for attempt in range(3):
        try:
            from ddgs import DDGS
            results = DDGS().text(query, max_results=8)
            if results:
                return "\n\n".join(
                    f"[{r.get('title')}]({r.get('href')})\n{r.get('body')}" for r in results
                )
        except Exception as e: 
            last_err = e
        time.sleep(2 * (attempt + 1))
    
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "srlimit": 8, "format": "json"},
            headers={"User-Agent": "gaia-agent/1.0"}, timeout=30,
        )
        hits = r.json()["query"]["search"]
        if hits:
            return "(DuckDuckGo failed; Wikipedia results)\n" + "\n\n".join(
                f"{h['title']}: {re.sub('<[^>]+>', '', h['snippet'])}" for h in hits
            )
    except Exception as e:  
        last_err = e
    return f"No results found (last error: {last_err}). Try different keywords or wikipedia_page."


@tool
def wikipedia_page(title: str, keyword: str = "") -> str:
    """Returns an English Wikipedia article as markdown, INCLUDING its tables (e.g. discographies).
    Args:
        title: exact article title, e.g. 'Mercedes Sosa'. Redirects are followed.
        keyword: optional word or phrase (e.g. 'Discography'); if given, the text returned starts just before its first occurrence.
    """
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "parse", "page": title, "prop": "text", "redirects": 1,
                    "format": "json", "formatversion": 2},
            headers={"User-Agent": "gaia-agent/1.0"}, timeout=30,
        )
        data = r.json()
        if "error" in data:
            return f"No Wikipedia article found for '{title}'. Try web_search to find the exact title."
        html = data["parse"]["text"]
        from markdownify import markdownify
        text = markdownify(html, heading_style="ATX")
        text = re.sub(r"\n{3,}", "\n\n", text)
    except Exception as e:  
        return f"Error fetching Wikipedia: {e}"
    if keyword:
        i = text.lower().find(keyword.lower())
        if i == -1:
            return f"Keyword '{keyword}' not found in '{title}'. Article length: {len(text)} chars."
        text = text[max(0, i - 300):]
    return text[:25000]


#Modelos de respaldo por si uno falla (el primero es el principal, los demás son de respaldo).
MODEL_CHAIN = [
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
    "gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-flash-lite-latest",
    "gemini-3-flash-preview", "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite",
]


def model_chain():
    chain = [f"gemini/{m}" for m in MODEL_CHAIN]
    forced = os.getenv("GEMINI_MODEL")
    if forced:
        chain = [forced] + [m for m in chain if m != forced]
    return chain


class ThrottledLiteLLMModel(LiteLLMModel):
    """Espera un mínimo entre llamadas (cuota gratis) y, si un modelo falla, prueba el siguiente.
    - Tope de tiempo REAL por llamada (litellm a veces tarda mucho más que su propio timeout).
    - Respuesta vacía = fallo (se pasa al siguiente modelo).
    - Un modelo que falla queda en 'cooldown' que crece si sigue fallando (5, 10, 20... hasta 30 min)."""

    def __init__(self, *args, min_interval: float = 6.5, chain=None, cooldown: float = 300,
                 hard_timeout: float = 85, max_total: float = 200, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_interval = min_interval
        self._last_call = 0.0
        self._chain = list(chain or [self.model_id])
        self._cooldown = cooldown
        self._hard_timeout = hard_timeout
        self._max_total = max_total  # tiempo máximo probando modelos en UNA llamada
        self._blocked = {}  
        self._fails = {}    

    def _candidates(self):
        now = time.time()
        alive = [m for m in self._chain if self._blocked.get(m, 0) != float("inf")]
        ready = [m for m in alive if self._blocked.get(m, 0) <= now]
        if ready:
            return ready
       
        return sorted(alive, key=lambda m: self._blocked[m])[:1]

    def _call(self, messages, kwargs):
        box = {}

        def work():
            try:
                box["out"] = LiteLLMModel.generate(self, messages, **kwargs)
            except BaseException as e:  
                box["err"] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()
        th.join(self._hard_timeout)
        if th.is_alive(): 
            raise TimeoutError(f"sin respuesta en {self._hard_timeout:.0f}s")
        if "err" in box:
            raise box["err"]
        return box["out"]

    def generate(self, messages, **kwargs):
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        start = time.time()
        tried = []
        try:
            for m in self._candidates():
                if tried and time.time() - start > self._max_total:
                    break
                self.model_id = m
                t0 = time.time()
                try:
                    out = self._call(messages, kwargs)
                    if not (getattr(out, "content", None) or getattr(out, "tool_calls", None)):
                        raise ValueError("respuesta vacía del modelo")
                    self._fails[m] = 0
                    self._blocked.pop(m, None)
                    if tried:
                        print(f"[RESPALDO] respondió {m}")
                    return out
                except Exception as e:  
                    msg = str(e)
                    dead = "NotFound" in type(e).__name__ or ("404" in msg and "not found" in msg.lower())
                    n = self._fails.get(m, 0) + 1
                    self._fails[m] = n
                    pause = 60 if isinstance(e, ValueError) else min(self._cooldown * 2 ** (n - 1), 1800)
                    self._blocked[m] = float("inf") if dead else time.time() + pause
                    tried.append(m)
                    print(f"\n[GEMINI ERROR {m} tras {time.time() - t0:.0f}s] "
                          f"{type(e).__name__}: {msg[:160]}\n")
            raise RuntimeError(f"Ningún modelo respondió (probados: {', '.join(tried) or 'ninguno'})")
        finally:
            self._last_call = time.time()


def build_agent(max_steps: int = 10, verbosity: int = 1) -> CodeAgent:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Falta GEMINI_API_KEY. Defínela con: conda env config vars set GEMINI_API_KEY=... "
            "(y reactiva el entorno) o ponla en un archivo .env"
        )
    chain = model_chain()
    timeout = float(os.getenv("GEMINI_TIMEOUT", "75"))
    model = ThrottledLiteLLMModel(
        model_id=chain[0],
        chain=chain,
        hard_timeout=timeout + 10,
        api_key=api_key,
        max_tokens=4096,
        num_retries=int(os.getenv("GEMINI_RETRIES", "0")),  
        timeout=timeout,  
        reasoning_effort=os.getenv("GEMINI_REASONING", "low"),
        min_interval=float(os.getenv("MIN_SECONDS_BETWEEN_CALLS", "6.5")),
    )
    
    # AGENT_FORMAT=structured -> el modelo responde JSON {thought, code}; AGENT_FORMAT=default -> <code>...</code>
    fmt = os.getenv("AGENT_FORMAT", "markdown").lower()
    fmt_kwargs = {}
    if fmt == "markdown":
        fmt_kwargs["code_block_tags"] = "markdown"
    elif fmt == "structured":
        fmt_kwargs["use_structured_outputs_internally"] = True
    return CodeAgent(
        model=model,
        tools=[web_search, wikipedia_page, VisitWebpageTool(), read_file, FinalAnswerTool()],
        **fmt_kwargs,
        # Librerías que el código generado por el modelo puede importar
        additional_authorized_imports=[
            "pandas", "numpy", "requests", "re", "json", "math", "datetime",
            "collections", "itertools", "statistics", "csv", "io", "bs4", "unicodedata",
        ],
        instructions=GAIA_RULES,
        max_steps=max_steps,
        planning_interval=None,  
        verbosity_level=verbosity,
    )



JUNK_EXACT = {"", "unknown", "none", "n/a", "na", "null", "unable", "no answer", "not found"}
MEDIA_EXT = (".mp3", ".wav", ".m4a", ".ogg", ".mp4", ".mov", ".avi", ".mkv")


def is_junk(ans) -> bool:
    """Respuestas que son una rendición, no una respuesta: no se guardan ni se envían."""
    a = str(ans).strip().lower().strip(".")
    return a in JUNK_EXACT or a.startswith(("i cannot", "i can't", "i am unable", "cannot ", "unable to"))


def needs_media(q) -> bool:
    """Video/audio: el agente no tiene herramientas para verlos; intentarlo solo gasta tokens."""
    text = q["question"].lower()
    f = (q.get("file_name") or "").lower()
    return "youtube.com" in text or "youtu.be" in text or f.endswith(MEDIA_EXT)


def clean_answer(text) -> str:
    s = str(text).strip()
    s = re.sub(r"^\s*final answer\s*:\s*", "", s, flags=re.I)
    s = s.strip().strip('"').strip("'").strip()
    if s.endswith(".") and not re.search(r"\d\.$", s):  
        s = s[:-1]
    return s


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_questions():
    r = requests.get(f"{API}/questions", timeout=30)
    r.raise_for_status()
    qs = r.json()
    save_json("questions.json", qs)
    return qs


def download_file(task_id, file_name):
    os.makedirs(DOWNLOADS, exist_ok=True)
    path = os.path.join(DOWNLOADS, file_name)
    if not os.path.exists(path):
        r = requests.get(f"{API}/files/{task_id}", timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def submit(answers):
    answers = {k: v for k, v in answers.items() if not is_junk(v)}
    user = os.getenv("HF_USERNAME")
    if not user:
        raise SystemExit("Falta HF_USERNAME (tu usuario de Hugging Face).")
    code_url = os.getenv("AGENT_CODE_URL") or f"https://huggingface.co/spaces/{user}/First_agent_template/tree/main"
    payload = {
        "username": user,
        "agent_code": code_url,
        "answers": [{"task_id": k, "submitted_answer": v} for k, v in answers.items()],
    }
    r = requests.post(f"{API}/submit", json=payload, timeout=120)
    print("HTTP", r.status_code)
    print(r.text)


MAX_CONSECUTIVE_FAILS = 4  


def build_prompt(q):
    """Devuelve (prompt, imágenes, ok). ok=False si la pregunta depende de un archivo que no se pudo bajar."""
    tid, prompt, images, fname = q["task_id"], q["question"], None, q.get("file_name")
    ok = True
    if fname:
        try:
            path = download_file(tid, fname)
            prompt += f"\n\nAttached file (local path): {path}"
            if path.lower().endswith(IMAGE_EXT):
                from PIL import Image
                images = [Image.open(path).convert("RGB")]
                prompt += "\n(The image is attached to this message; look at it.)"
            else:
                prompt += "\nUse the read_file tool to read it."
        except Exception as e: 
            ok = False
            prompt += f"\n\n(The attached file '{fname}' could not be downloaded: {e})"
    return prompt, images, ok


def run_gaia(args):
    answers = load_json(ANSWERS_FILE, {})
    questions = fetch_questions()
    if args.task_id:
        questions = [q for q in questions if q["task_id"] == args.task_id]
    if args.limit:
        questions = questions[: args.limit]

    
    pending = [q for q in questions if args.force or is_junk(answers.get(q["task_id"], ""))]
    done = len(questions) - len(pending)
    print(f"Preguntas: {len(questions)} | ya respondidas (se saltan): {done} | pendientes: {len(pending)}")

    agent = None  
    fails = 0
    skipped = []
    no_tools = []
    for i, q in enumerate(pending, 1):
        tid = q["task_id"]
        if needs_media(q) and not args.try_media:
            print(f"[{i}/{len(pending)}] {tid}: requiere ver/oír un video o audio -> omitida (0 tokens)")
            no_tools.append(tid)
            continue
        prompt, images, ok = build_prompt(q)
        if not ok:
            print(f"[{i}/{len(pending)}] {tid}: su archivo adjunto no se pudo descargar -> omitida (0 tokens)")
            no_tools.append(tid)
            continue
        if agent is None:
            agent = build_agent()
        print(f"\n===== [{i}/{len(pending)}] {tid} =====\n{q['question']}\n")
        try:
            ans = clean_answer(agent.run(prompt, images=images))
            if is_junk(ans):
                raise ValueError(f"el agente no encontró respuesta ({ans!r})")
        except Exception as e: 
            print(f"SIN RESPUESTA en {tid}, paso a la siguiente: {e}")
            skipped.append(tid)
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                print(f"\n{fails} preguntas seguidas sin respuesta: los modelos parecen caídos. "
                      "Corto aquí; lo ya respondido está guardado. Reintenta más tarde con el mismo comando.")
                break
            continue  
        fails = 0
        answers[tid] = ans
        save_json(ANSWERS_FILE, answers)
        print(f"-> respuesta: {ans!r}")
        time.sleep(args.pause)

    total_done = sum(1 for q in questions if not is_junk(answers.get(q["task_id"], "")))
    print(f"\nRespondidas: {total_done}/{len(questions)} (guardadas en {ANSWERS_FILE})")
    if no_tools:
        print(f"Omitidas sin gastar tokens ({len(no_tools)}): video/audio o archivo no disponible.")
    if skipped:
        print(f"Sin respuesta ({len(skipped)}): vuelve a correr el mismo comando para reintentar solo esas.")
    if args.submit:
        if answers:
            submit(answers)
        else:
            print("No hay respuestas guardadas; no envío nada.")


def ping():
    """Prueba cada modelo de la cadena con una llamada mínima y muestra cuáles responden."""
    print(f"{'modelo':34} {'resultado':12} segundos")
    for m in model_chain():
        t0 = time.time()
        try:
            litellm.completion(
                model=m, messages=[{"role": "user", "content": "Say hi in one word."}],
                api_key=os.getenv("GEMINI_API_KEY"), max_tokens=512,
                reasoning_effort=os.getenv("GEMINI_REASONING", "low"), timeout=25, num_retries=0,
            )
            res = "OK"
        except Exception as e: 
            res = type(e).__name__[:12]
        print(f"{m:34} {res:12} {time.time() - t0:.1f}")


def main():
    ap = argparse.ArgumentParser(description="Agente GAIA con Gemini (todo en app.py)")
    ap.add_argument("--gaia", action="store_true", help="correr las preguntas del reto")
    ap.add_argument("--submit", action="store_true", help="enviar al scoring al terminar")
    ap.add_argument("--submit-only", action="store_true", help="enviar lo guardado en answers.json")
    ap.add_argument("--limit", type=int, default=None, help="solo las N primeras preguntas")
    ap.add_argument("--task-id", default=None, help="solo esta pregunta")
    ap.add_argument("--force", action="store_true", help="repetir aunque ya haya respuesta")
    ap.add_argument("--try-media", action="store_true", help="intentar también las preguntas de video/audio (gasta tokens)")
    ap.add_argument("--ping", action="store_true", help="probar solo la clave y el modelo de Gemini")
    ap.add_argument("--pause", type=float, default=3.0, help="segundos de pausa entre preguntas")
    args = ap.parse_args()

    if args.ping:
        ping()
        return
    if args.submit_only:
        submit(load_json(ANSWERS_FILE, {}))
    elif args.gaia or args.submit:
        run_gaia(args)
    else:
        GradioUI(build_agent()).launch()  


if __name__ == "__main__":
    main()
