"""
Чат-бот про программы и гранты фонда Ш. Есенова.

Главный принцип: бот НЕ выдумывает. Он отвечает строго по данным,
собранным в папке data/ (скриптом scrape.py). Если в данных нет ответа
(дедлайн, сумма, требование) — честно говорит, что точной информации нет,
и советует уточнить у фонда. Выдуманная цифра = провал, потому что человек
примет по ней реальное решение.

Запуск:  streamlit run app.py
"""

import os
import re
import time
from pathlib import Path

import numpy as np
import requests
import streamlit as st

# --- настройки LLM gemma4 ---
LLM_URL = "https://llm.alem.ai/v1/chat/completions"
LLM_MODEL = "gemma4"

# --- настройки embedding-модели (RAG) ---
EMB_URL = "https://llm.alem.ai/v1/embeddings"
EMB_MODEL = "text-1024"
EMB_DIM = 1024

# Сколько ближайших chunk-ов класть в промпт и порог отсечения.
# Если максимальное косинусное сходство ниже порога — считаем, что в данных
# нет релевантной информации, и НЕ зовём gemma4 (честно отвечаем «не знаю»).
# Порог 0.45 подобран на реальных embeddings text-1024: релевантные вопросы
# давали сходство 0.55–0.85, посторонние — 0.15–0.39. 0.45 лежит в разрыве.
# ВРЕМЕННО отключён (0) для проверки — вернуть на 0.45 после отладки.
TOP_K = 8
SIM_THRESHOLD = 0.0

# Ключи НЕ хранятся в коде (чтобы не попасть в git). Берём из переменной
# окружения или из .streamlit/secrets.toml.
#   локально:  export GEMMA_KEY="sk-..." ; export EMB_KEY="sk-..."
def _get_key(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


LLM_KEY = _get_key("GEMMA_KEY")
EMB_KEY = _get_key("EMB_KEY")

DATA_DIR = Path(__file__).parent / "data"

# Системная инструкция — здесь и заложена защита от выдумывания.
SYSTEM_PROMPT = """Ты — ассистент Научно-образовательного фонда им. Шахмардана Есенова.
Ты помогаешь людям с вопросами о программах и грантах фонда.

СТРОГИЕ ПРАВИЛА (нарушать нельзя):
1. Отвечай ТОЛЬКО на основе данных в разделе «ДАННЫЕ ФОНДА» ниже.
   Никакие свои знания, догадки или предположения использовать НЕЛЬЗЯ.
2. Если в данных нет ответа на вопрос (например, точный дедлайн, сумма
   гранта, требование к документам) — честно скажи: «Точной информации
   об этом в моих данных нет» и посоветуй уточнить напрямую у фонда
   (сайт yessenovfoundation.org). НИКОГДА не придумывай даты, суммы,
   проценты или условия. Лучше сказать «не знаю», чем ошибиться.
3. Если данные есть, но они могут быть устаревшими (например, относятся
   к прошлому году) — обязательно предупреди об этом.
4. По возможности указывай, из какой программы взят ответ
   (название программы / файла).
5. Отвечай на том языке, на котором задан вопрос (русский / казахский).

Помни: по твоему ответу человек примет реальное решение. Точность важнее
полноты. Выдуманная цифра недопустима."""


def embed(text, retries=4):
    """Получить embedding одного текста через OpenAI-совместимый API.

    С ретраями: эндпоинт иногда отдаёт временные 502/503/таймаут, а при
    построении индекса делается много запросов подряд.
    """
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                EMB_URL,
                json={"model": EMB_MODEL, "input": text},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {EMB_KEY}",
                },
                timeout=60,
            )
            resp.raise_for_status()
            vec = resp.json()["data"][0]["embedding"]
            return np.asarray(vec, dtype=np.float32)
        except Exception as e:
            last_err = e
            # на 429 (rate limit) ждём дольше, на прочих ошибках — короткий backoff
            is_429 = getattr(getattr(e, "response", None), "status_code", None) == 429
            base = 6.0 if is_429 else 1.5
            time.sleep(base * (attempt + 1))
    raise RuntimeError(f"embeddings API недоступен после {retries} попыток: {last_err}")


def split_into_chunks(text, max_chars=500, overlap_lines=2):
    """Разбить текст программы на chunks по абзацам (пустая строка = граница).

    Очень длинные абзацы (например, текст PDF) дополнительно режем по строкам,
    чтобы chunk не был гигантским и поиск оставался точным.

    max_chars небольшой (500) намеренно: в исходных данных смысловые блоки
    (срок подачи, требования, документы) идут подряд без пустых строк, поэтому
    весь блок — один мегаабзац. При крупном chunk-е важный факт (например,
    «срок подачи: с 26 марта по 28 апреля») тонул в массе текста про документы,
    и его embedding переставал отвечать на вопрос про дедлайн. Мелкий chunk
    делает факт «весомее» в своём векторе → поиск по дедлайну попадает в цель.

    overlap_lines: последние строки предыдущего chunk-а переносим в начало
    следующего, чтобы факт на границе (заголовок ↔ значение) не разрывался.
    """
    chunks = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if len(para) < 20:  # пропускаем мусорные обрывки
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        # длинный абзац -> копим строки до лимита, с перекрытием между chunk-ами
        buf_lines, buf_len = [], 0
        for line in para.splitlines():
            if buf_len + len(line) + 1 > max_chars and buf_lines:
                chunks.append("\n".join(buf_lines).strip())
                buf_lines = buf_lines[-overlap_lines:]  # хвост -> в следующий chunk
                buf_len = sum(len(l) + 1 for l in buf_lines)
            buf_lines.append(line)
            buf_len += len(line) + 1
        if buf_lines and "\n".join(buf_lines).strip():
            chunks.append("\n".join(buf_lines).strip())
    return chunks


@st.cache_resource(show_spinner="Строю индекс (считаю embeddings)…")
def build_index():
    """Прочитать data/*.txt, нарезать на chunks, посчитать embeddings.

    Считается ОДИН раз (cache_resource). Возвращает:
      chunks  — список dict {text, source}
      matrix  — нормированная матрица embeddings [N, EMB_DIM]
      names   — список имён программ
    """
    files = sorted(DATA_DIR.glob("*.txt"))
    chunks = []
    names = []
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue
        names.append(f.stem)
        for piece in split_into_chunks(text):
            chunks.append({"text": piece, "source": f.stem})

    # embeddings всех chunk-ов (лёгкий троттлинг, чтобы не словить rate limit
    # на первой сборке индекса — потом всё кэшируется и больше не считается)
    vectors = np.zeros((len(chunks), EMB_DIM), dtype=np.float32)
    for i, ch in enumerate(chunks):
        vectors[i] = embed(ch["text"])
        time.sleep(0.1)

    # нормируем строки -> косинусное сходство = скалярное произведение
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = vectors / norms
    return chunks, matrix, names


def detect_program(question, history, program_names):
    """Определить, о какой программе идёт речь сейчас, по названию в тексте.

    Сканируем от свежего к старому: текущий вопрос -> прошлые вопросы
    пользователя -> последний ответ бота. Первое совпадение названия программы
    и есть «программа в фокусе».

    Зачем не эмбеддинг: тематические вопросы («какие документы», «дедлайн»)
    одинаково похожи на соответствующие чанки ВО ВСЕХ программах, и поиск по
    смыслу смешивал бы программы. Названия же различимы, а в диалоге программа
    почти всегда названа в предыдущем вопросе — строковый матч надёжнее и
    предсказуемее для пользователя.
    """
    if not program_names:
        return None
    candidates = [question]
    if history:
        # прошлые вопросы пользователя (свежие — раньше в списке = выше приоритет)
        candidates += [m["content"] for m in reversed(history) if m["role"] == "user"]
        # запасной вариант: последний ответ бота (если программу назвал он)
        asst = [m["content"] for m in history if m["role"] == "assistant"]
        if asst:
            candidates.append(asst[-1])
    for text in candidates:
        low = text.lower()
        for name in program_names:
            if name.lower() in low:
                return name
    return None


def retrieve(question, chunks, matrix, history=None, program_names=None):
    """Найти top-K chunk-ов по косинусному сходству. Вернуть (список, max_sim, focus).

    Сходство считаем по ТЕМЕ голого вопроса («дедлайн», «документы»). Если по
    диалогу понятно, о какой программе речь (focus), — ОГРАНИЧИВАЕМ поиск
    чанками этой программы. Иначе тематические чанки собирались бы из всех
    программ сразу, и бот отвечал «вот документы для всех программ».
    """
    q = embed(question)
    q = q / (np.linalg.norm(q) or 1.0)
    sims = matrix @ q  # косинусное сходство по теме вопроса

    focus = detect_program(question, history, program_names)
    if focus is not None:
        # прячем чужие программы: их сходства -> -1, в top-K попадут только
        # чанки программы в фокусе, ранжированные по теме вопроса
        mask = np.array([c["source"] == focus for c in chunks])
        if mask.any():
            sims = np.where(mask, sims, -1.0)

    top_idx = np.argsort(-sims)[:TOP_K]
    results = [
        {**chunks[i], "score": float(sims[i])} for i in top_idx
    ]
    max_sim = float(sims[top_idx[0]]) if len(top_idx) else 0.0
    return results, max_sim, focus


def ask_gemma(knowledge, history, question, program_names=None):
    """Собрать запрос (инструкция + данные + вопрос) и вызвать gemma4."""
    # Полный список программ кладём в промпт всегда. Ретривер достаёт только
    # top-K фрагментов, поэтому на вопрос «перечисли все программы» модель
    # видела бы лишь 3-8 источников. Список имён — копейки по токенам, зато
    # обзорные вопросы («что есть в фонде») начинают отвечаться корректно.
    programs_block = ""
    if program_names:
        programs_block = (
            "\n\nВСЕ ПРОГРАММЫ ФОНДА (полный список названий): "
            + ", ".join(program_names)
        )

    system_content = (
        SYSTEM_PROMPT
        + programs_block
        + "\n\n=== ДАННЫЕ ФОНДА (единственный разрешённый источник) ===\n"
        + knowledge
        + "\n=== КОНЕЦ ДАННЫХ ==="
    )

    messages = [{"role": "system", "content": system_content}]
    # короткая память диалога
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0,  # минимум «фантазии»
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_KEY}",
    }

    resp = requests.post(LLM_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ----------------- интерфейс -----------------
st.set_page_config(page_title="Чат-бот фонда Есенова", page_icon="🎓")
st.title("🎓 Чат-бот фонда Ш. Есенова")
st.caption(
    "Отвечаю только по данным с сайта фонда. Если чего-то не знаю — так и скажу, "
    "а не придумаю. Важные суммы и сроки всегда перепроверяйте у фонда."
)

if not list(DATA_DIR.glob("*.txt")):
    st.error("Нет данных в папке data/. Сначала запустите: python3 scrape.py")
    st.stop()

if not LLM_KEY or not EMB_KEY:
    st.error(
        "Не заданы API-ключи. Установите переменные окружения GEMMA_KEY и EMB_KEY "
        "(export GEMMA_KEY=\"sk-...\" ; export EMB_KEY=\"sk-...\") "
        "или добавьте их в .streamlit/secrets.toml."
    )
    st.stop()

chunks, matrix, program_names = build_index()

with st.sidebar:
    st.subheader("Загруженные программы")
    for name in program_names:
        st.write(f"• {name}")
    st.caption(f"Индекс: {len(chunks)} фрагментов, top-{TOP_K}, порог {SIM_THRESHOLD}")

if "history" not in st.session_state:
    st.session_state.history = []

# показать историю
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ввод пользователя
question = st.chat_input("Спросите про программу, грант, дедлайн…")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Ищу ответ в данных фонда…"):
            try:
                found, max_sim, focus = retrieve(
                    question,
                    chunks,
                    matrix,
                    st.session_state.history[:-1],
                    program_names,
                )

                # DEBUG: показать найденные чанки и их similarity-баллы
                focus_label = focus or "не определена (поиск по всем)"
                with st.expander(
                    f"🔍 debug: программа={focus_label} · "
                    f"найденные чанки (max_sim={max_sim:.3f})"
                ):
                    for rank, c in enumerate(found, 1):
                        st.markdown(
                            f"**#{rank} · score={c['score']:.3f} · источник: {c['source']}**"
                        )
                        st.text(c["text"])

                if max_sim < SIM_THRESHOLD:
                    # ничего достаточно похожего не нашлось — не зовём gemma4
                    answer = (
                        "В моих данных нет информации по этому вопросу. "
                        "Рекомендую уточнить напрямую у фонда "
                        "(сайт yessenovfoundation.org)."
                    )
                else:
                    # в промпт кладём ТОЛЬКО найденные фрагменты с источником
                    context = "\n\n".join(
                        f"### ПРОГРАММА: {c['source']}\n{c['text']}" for c in found
                    )
                    answer = ask_gemma(
                        context,
                        st.session_state.history[:-1],
                        question,
                        program_names,
                    )
            except Exception as e:
                answer = (
                    "Не удалось получить ответ от модели. "
                    f"Техническая ошибка: {e}"
                )
        st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})
