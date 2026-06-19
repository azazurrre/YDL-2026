"""
Чат-бот про программы и гранты фонда Ш. Есенова.

Главный принцип: бот НЕ выдумывает. Он отвечает строго по данным,
собранным в папке data/ (скриптом scrape.py). Если в данных нет ответа
(дедлайн, сумма, требование) — честно говорит, что точной информации нет,
и советует уточнить у фонда. Выдуманная цифра = провал, потому что человек
примет по ней реальное решение.

Запуск:  streamlit run app.py
"""

import hashlib
import json
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
# Порог низкий (0.30) намеренно: это лишь дешёвый предохранитель против
# заведомо посторонних вопросов (где звать gemma бессмысленно). Основной
# отказ «в данных нет» обеспечивает системный промпт — он надёжно работает
# даже на пограничных вопросах. Калибровка на реальных embeddings text-1024
# (чанки 500 симв. + маскирование по программе): валидные вопросы с ответом
# дают 0.75–0.87, посторонние/без-ответа — около 0.35 и ниже. 0.30 лежит ниже
# разрыва, поэтому не режет валидные ответы, а спорное отдаёт на суд gemma.
TOP_K = 8
SIM_THRESHOLD = 0.30

# Показывать ли debug-панель с найденными чанками. Перед показом
# пользователям / деплоем поставить False.
DEBUG = True

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

# --- настройки отправки email (MailerSend) ---
# Письма шлём ТОЛЬКО на свой собственный ADMIN_EMAIL и ТОЛЬКО по кнопке —
# никогда в цикле. Иначе один баг разошлёт пачку писем и испортит репутацию
# домена-отправителя у Gmail (вся почта домена начнёт уходить в спам).
MAILERSEND_KEY = _get_key("MAILERSEND_KEY")
ADMIN_EMAIL = _get_key("ADMIN_EMAIL")
# Отправитель ДОЛЖЕН быть на домене, верифицированном в твоём аккаунте
# MailerSend (иначе отправка падает). По умолчанию — домен курса; если у тебя
# свой trial-домен (вида test-xxxx.mlsender.net), задай FROM_EMAIL в secrets.
FROM_EMAIL = _get_key("FROM_EMAIL") or "info@app.commit.kz"
FROM_NAME = "Yessenov Data Lab"

DATA_DIR = Path(__file__).parent / "data"
# Куда складываем посчитанные embeddings, чтобы не пересчитывать при каждом
# перезапуске. Кэш привязан к содержимому data/*.txt и параметрам нарезки —
# меняются данные или нарезка => индекс считается заново.
CACHE_DIR = Path(__file__).parent / ".cache"

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


def _index_signature(texts):
    """Отпечаток корпуса: содержимое всех файлов + модель + параметры нарезки.

    Если поменялись данные, embedding-модель или дефолты split_into_chunks —
    отпечаток меняется, и дисковый кэш считается невалидным (пересчёт).
    """
    h = hashlib.sha256()
    chunk_params = repr(split_into_chunks.__defaults__)  # (max_chars, overlap)
    h.update(f"v1|{EMB_MODEL}|{EMB_DIM}|{chunk_params}".encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@st.cache_resource(show_spinner="Строю индекс (считаю embeddings)…")
def build_index():
    """Прочитать data/*.txt, нарезать на chunks, посчитать embeddings.

    Двухуровневый кэш:
      1. @st.cache_resource — держит готовый индекс в памяти на время сессии.
      2. .cache/index_<отпечаток>.npy — сохраняет матрицу embeddings на диск,
         поэтому при перезапуске процесса она грузится мгновенно, без запросов
         к embedding-API. Пересчёт только если изменились данные/нарезка/модель.

    Возвращает:
      chunks  — список dict {text, source}
      matrix  — нормированная матрица embeddings [N, EMB_DIM]
      names   — список имён программ
    """
    files = sorted(DATA_DIR.glob("*.txt"))
    chunks = []
    names = []
    texts = []
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue
        names.append(f.stem)
        texts.append(text)
        for piece in split_into_chunks(text):
            chunks.append({"text": piece, "source": f.stem})

    # nарезка детерминирована, поэтому достаточно кэшировать только матрицу:
    # chunks восстанавливаются из тех же текстов в том же порядке.
    sig = _index_signature(texts)
    cache_file = CACHE_DIR / f"index_{sig}.npy"
    if cache_file.exists():
        matrix = np.load(cache_file)
        if matrix.shape == (len(chunks), EMB_DIM):  # защита от рассинхрона
            return chunks, matrix, names

    # embeddings всех chunk-ов (лёгкий троттлинг, чтобы не словить rate limit
    # на первой сборке индекса — потом кэшируется на диск и больше не считается)
    vectors = np.zeros((len(chunks), EMB_DIM), dtype=np.float32)
    for i, ch in enumerate(chunks):
        vectors[i] = embed(ch["text"])
        time.sleep(0.1)

    # нормируем строки -> косинусное сходство = скалярное произведение
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = vectors / norms

    # сохраняем на диск; старые отпечатки чистим, чтобы кэш не разрастался
    CACHE_DIR.mkdir(exist_ok=True)
    for old in CACHE_DIR.glob("index_*.npy"):
        old.unlink()
    np.save(cache_file, matrix)
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


def summarize_dialog(history):
    """Короткое саммари диалога через gemma4 — для письма администратору."""
    transcript = "\n".join(
        f"{'Пользователь' if m['role'] == 'user' else 'Бот'}: {m['content']}"
        for m in history
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Сделай короткое саммари диалога пользователя с чат-ботом фонда "
                "Есенова для администратора фонда. Укажи: какие программы/гранты "
                "интересовали, какой главный запрос, оставил ли пользователь "
                "заявку или контакт. 3–6 предложений, по-русски, по делу, без воды."
            ),
        },
        {"role": "user", "content": transcript},
    ]
    payload = {"model": LLM_MODEL, "messages": messages, "temperature": 0.2}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_KEY}",
    }
    resp = requests.post(LLM_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def send_summary_email(summary):
    """Отправить саммари ТОЛЬКО на ADMIN_EMAIL (себе) через MailerSend.

    Вызывается строго по кнопке (явное действие пользователя), не в цикле.
    Возвращает message_id при успехе.
    """
    from mailersend import MailerSendClient, EmailBuilder

    ms = MailerSendClient(api_key=MAILERSEND_KEY)
    html = "<h2>Саммари разговора (чат-бот фонда Есенова)</h2>" + "".join(
        f"<p>{line}</p>" for line in summary.splitlines() if line.strip()
    )
    email = (
        EmailBuilder()
        .from_email(FROM_EMAIL, FROM_NAME)
        .to_many([{"email": ADMIN_EMAIL, "name": "Admin"}])
        .subject("Новая заявка/диалог из чата фонда")
        .html(html)
        .text(summary)
        .build()
    )
    response = ms.emails.send(email)
    # MailerSend возвращает id письма в заголовке x-message-id (в теле его нет)
    headers = getattr(response, "headers", None) or {}
    return headers.get("x-message-id") or headers.get("X-Message-Id")


def detect_application(history):
    """Решение МОДЕЛИ: оставил ли пользователь заявку/запрос для администратора.

    Возвращает (notify: bool, summary: str). Это «осознанное решение модели»
    из задания — отдельный строгий вызов, который классифицирует диалог и,
    если это заявка/запрос/оставленный контакт, готовит саммари.

    Fail-safe: при любой неоднозначности или ошибке парсинга -> notify=False.
    Лучше НЕ отправить, чем разослать лишнее и испортить репутацию домена.
    """
    transcript = "\n".join(
        f"{'Пользователь' if m['role'] == 'user' else 'Бот'}: {m['content']}"
        for m in history
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — фильтр уведомлений администратора фонда Есенова. Определи, "
                "оставил ли ПОЛЬЗОВАТЕЛЬ в диалоге ЗАЯВКУ или конкретный ЗАПРОС, "
                "о котором стоит уведомить администратора. notify=true ТОЛЬКО "
                "если пользователь: хочет подать заявку на программу, просит "
                "связаться с ним, оставил контакт (email/телефон), просит "
                "записать/зарегистрировать его. Обычные информационные вопросы "
                "(что такое программа, дедлайн, документы) -> notify=false.\n"
                "Ответь СТРОГО одним JSON-объектом без пояснений и без markdown:\n"
                '{"notify": true|false, "summary": "<2-4 предложения для '
                'администратора: кто, что хочет, оставленные контакты>"}'
            ),
        },
        {"role": "user", "content": transcript},
    ]
    payload = {"model": LLM_MODEL, "messages": messages, "temperature": 0}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_KEY}",
    }
    try:
        resp = requests.post(LLM_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # снимаем возможные ```json … ``` обёртки
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
        # берём первый JSON-объект из ответа
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        data = json.loads(match.group(0) if match else content)
        return bool(data.get("notify")), str(data.get("summary", "")).strip()
    except Exception:
        return False, ""


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

if "history" not in st.session_state:
    st.session_state.history = []
# флаг «уже уведомили администратора в этой сессии» — предохранитель от
# повторных писем (правило «никогда в цикле»)
if "notified" not in st.session_state:
    st.session_state.notified = False

with st.sidebar:
    st.subheader("Загруженные программы")
    for name in program_names:
        st.write(f"• {name}")
    st.caption(f"Индекс: {len(chunks)} фрагментов, top-{TOP_K}, порог {SIM_THRESHOLD}")

    # --- Отправка саммари администратору (шаг «чат -> агент») ---
    st.divider()
    st.subheader("📧 Администратору")
    if not (MAILERSEND_KEY and ADMIN_EMAIL):
        st.caption(
            "Отправка выключена: задайте MAILERSEND_KEY и ADMIN_EMAIL "
            "в .streamlit/secrets.toml."
        )
    else:
        st.caption(f"Саммари разговора уйдёт на {ADMIN_EMAIL} (только вам).")
        # Авто-режим: модель сама решает, заявка ли это, и шлёт письмо.
        # Можно выключить для демо ручной кнопки.
        st.checkbox(
            "Авто-отправка при заявке/запросе",
            value=True,
            key="auto_notify",
            help="Бот сам отправит саммари, если решит, что вы оставили заявку. "
            "Не чаще одного письма за сессию.",
        )
        if st.session_state.notified:
            st.caption("✓ Администратор уже уведомлён в этой сессии.")
        # Кнопка = явное действие. Срабатывает один раз на клик, не в цикле.
        if st.button(
            "Отправить саммари вручную",
            disabled=not st.session_state.history,
            use_container_width=True,
        ):
            with st.spinner("Готовлю саммари и отправляю…"):
                try:
                    summary = summarize_dialog(st.session_state.history)
                    msg_id = send_summary_email(summary)
                    st.session_state.notified = True
                    st.success(f"Отправлено на {ADMIN_EMAIL}.")
                    if msg_id:
                        st.caption(f"message_id: {msg_id}")
                    with st.expander("Что отправили"):
                        st.write(summary)
                except Exception as e:
                    st.error(f"Не удалось отправить: {e}")

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
                if DEBUG:
                    focus_label = focus or "не определена (поиск по всем)"
                    with st.expander(
                        f"🔍 debug: программа={focus_label} · "
                        f"найденные чанки (max_sim={max_sim:.3f})"
                    ):
                        for rank, c in enumerate(found, 1):
                            st.markdown(
                                f"**#{rank} · score={c['score']:.3f} · "
                                f"источник: {c['source']}**"
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

    # --- Агентское действие: модель сама решает, заявка ли это, и шлёт письмо ---
    # Предохранители (правило «никогда в цикле»):
    #   • срабатывает только внутри обработки нового вопроса (не на каждом rerun);
    #   • не чаще одного письма за сессию (флаг notified);
    #   • при сомнении/ошибке detect_application возвращает notify=False.
    if (
        st.session_state.get("auto_notify")
        and not st.session_state.notified
        and MAILERSEND_KEY
        and ADMIN_EMAIL
    ):
        try:
            notify, summary = detect_application(st.session_state.history)
            if notify and summary:
                send_summary_email(summary)
                st.session_state.notified = True
                st.info(
                    "📧 Похоже на заявку — отправил саммари администратору "
                    f"({ADMIN_EMAIL})."
                )
                with st.expander("Что отправлено администратору"):
                    st.write(summary)
        except Exception as e:
            # сбой почты не должен ломать чат; в debug показываем причину
            if DEBUG:
                st.warning(f"авто-уведомление не сработало: {e}")
