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
from pathlib import Path

import requests
import streamlit as st

# --- настройки LLM gemma4 ---
LLM_URL = "https://llm.alem.ai/v1/chat/completions"
LLM_MODEL = "gemma4"

# Ключ НЕ хранится в коде (чтобы не попал в git). Берём из переменной
# окружения GEMMA_KEY или из .streamlit/secrets.toml (st.secrets["GEMMA_KEY"]).
#   локально:  export GEMMA_KEY="sk-..."   перед  streamlit run app.py
def _get_key():
    if os.environ.get("GEMMA_KEY"):
        return os.environ["GEMMA_KEY"]
    try:
        return st.secrets.get("GEMMA_KEY", "")
    except Exception:
        return ""


LLM_KEY = _get_key()

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


@st.cache_data(show_spinner=False)
def load_knowledge():
    """Прочитать все .txt из data/ и склеить в один контекст.

    Возвращает (текст_контекста, список_имён_программ).
    """
    files = sorted(DATA_DIR.glob("*.txt"))
    parts = []
    names = []
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue
        names.append(f.stem)
        parts.append(f"### ПРОГРАММА: {f.stem}\n{text}")
    context = "\n\n".join(parts)
    return context, names


def ask_gemma(knowledge, history, question):
    """Собрать запрос (инструкция + данные + вопрос) и вызвать gemma4."""
    system_content = (
        SYSTEM_PROMPT
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

knowledge, program_names = load_knowledge()

if not knowledge:
    st.error("Нет данных в папке data/. Сначала запустите: python3 scrape.py")
    st.stop()

if not LLM_KEY:
    st.error(
        "Не задан API-ключ. Установите переменную окружения GEMMA_KEY "
        "(export GEMMA_KEY=\"sk-...\") или добавьте её в .streamlit/secrets.toml."
    )
    st.stop()

with st.sidebar:
    st.subheader("Загруженные программы")
    for name in program_names:
        st.write(f"• {name}")

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
                answer = ask_gemma(
                    knowledge, st.session_state.history[:-1], question
                )
            except Exception as e:
                answer = (
                    "Не удалось получить ответ от модели. "
                    f"Техническая ошибка: {e}"
                )
        st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})
