"""
Сбор данных с сайта фонда Ш. Есенова для чат-бота.

Что делает скрипт:
1. Берёт список программ со страницы /about-us/programs/ (имя + ссылка).
2. Для каждой программы при необходимости доходит до КОНЕЧНОЙ страницы
   (например, у "Стажировок" — до программы текущего года 2026, а не до архива).
3. Достаёт ТОЛЬКО основной текст статьи (блок .post), отрезая меню, список
   партнёров и футер с контактами.
4. Скачивает прилагающиеся файлы (PDF и т.п.) и вытаскивает из PDF текст —
   именно там лежат реальные суммы грантов и дедлайны.
5. Сохраняет один .txt-файл на программу в data/. В начале файла и перед
   каждым блоком стоит строка ИСТОЧНИК с URL — чтобы факты были проверяемы
   и бот не выдумывал.

Запуск:  python3 scrape.py
"""

import io
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

BASE = "https://yessenovfoundation.org"
PROGRAMS_URL = f"{BASE}/about-us/programs/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; YessenovBot/1.0; educational project)"}

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FILES_DIR = DATA_DIR / "files"

FILE_EXTS = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip")


def get_soup(url):
    """Загрузить страницу и вернуть BeautifulSoup (или None при ошибке)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"   [!] не удалось загрузить {url}: {e}")
        return None


def clean_url(href):
    """Убрать хвост / и параметры для сравнения путей."""
    return href.split("?")[0].split("#")[0].rstrip("/")


def is_ru_program_link(href):
    """Только русскоязычные ссылки внутри раздела программ."""
    return "/about-us/programs/" in href and not any(
        p in href for p in ("/en/", "/kk/", "/ru/")
    )


def list_programs():
    """Список программ с главной страницы: [(имя, url), ...]."""
    soup = get_soup(PROGRAMS_URL)
    if not soup:
        raise SystemExit("Не удалось загрузить страницу программ")
    content = soup.select_one(".content") or soup.select_one(".post") or soup
    programs = []
    seen = set()
    for a in content.find_all("a", href=True):
        href = urljoin(BASE, a["href"])
        if not is_ru_program_link(href):
            continue
        if clean_url(href) == clean_url(PROGRAMS_URL):
            continue
        name = a.get_text(" ", strip=True)
        key = clean_url(href)
        if name and key not in seen:
            seen.add(key)
            programs.append((name, href))
    return programs


def find_final_page(landing_url, soup):
    """
    Если у программы есть более глубокие под-страницы (архив по годам),
    выбрать самую свежую (с максимальным годом). Иначе вернуть None.
    Берём только ОДНУ свежую страницу, чтобы не смешивать дедлайны разных лет.
    """
    base = clean_url(landing_url)
    container = soup.select_one(".post") or soup.select_one(".content") or soup
    candidates = {}
    for a in container.find_all("a", href=True):
        href = clean_url(urljoin(BASE, a["href"]))
        if not is_ru_program_link(href):
            continue
        if href.startswith(base + "/") and href != base:
            # под-страница ровно на один уровень глубже
            tail = href[len(base) + 1:]
            if "/" in tail:
                continue
            text = a.get_text(" ", strip=True)
            years = re.findall(r"(20\d{2})", tail + " " + text)
            year = max(int(y) for y in years) if years else 0
            candidates[href] = year
    if not candidates:
        return None
    # самая свежая по году, при равенстве — длиннее slug (обычно конкретнее)
    best = max(candidates, key=lambda u: (candidates[u], len(u)))
    return best


def resolve_file_url(href):
    """
    Ссылки на файлы обёрнуты: ?download=1&kccpid=..&kcccount=<реальный_url>.
    Возвращаем реальный URL файла.
    """
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    if "kcccount" in qs:
        return qs["kcccount"][0]
    return href


def extract_post(soup):
    """Вернуть (заголовок, чистый_текст, [(имя_файла, url_файла), ...]) из .post."""
    post = soup.select_one(".post")
    if not post:
        return None, "", []

    # убрать кнопки шаринга / скрипты / стили внутри статьи
    for junk in post.select("script, style, .ya-share2, .breadcrumbs, noscript"):
        junk.decompose()

    title_tag = post.find(["h1", "h2"])
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    # файлы
    files = []
    for a in post.find_all("a", href=True):
        real = resolve_file_url(urljoin(BASE, a["href"]))
        if real.lower().split("?")[0].endswith(FILE_EXTS):
            name = a.get_text(" ", strip=True) or real.rsplit("/", 1)[-1]
            files.append((name, real))

    # текст: построчно, без пустых строк
    lines = [ln.strip() for ln in post.get_text("\n").splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return title, text, files


def download_and_read_pdf(url, dest_dir, prefix):
    """
    Скачать файл в dest_dir. Если PDF — вернуть извлечённый текст,
    иначе вернуть "" (но файл всё равно сохраняем).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = url.rsplit("/", 1)[-1].split("?")[0] or "file"
    fname = re.sub(r"[^\w.\-]+", "_", fname)
    dest = dest_dir / f"{prefix}__{fname}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
    except Exception as e:
        print(f"   [!] не скачался файл {url}: {e}")
        return ""

    if dest.suffix.lower() != ".pdf":
        return ""
    try:
        reader = PdfReader(io.BytesIO(dest.read_bytes()))
        parts = [(p.extract_text() or "").strip() for p in reader.pages]
        return "\n".join(p for p in parts if p)
    except Exception as e:
        print(f"   [!] не прочитался PDF {dest.name}: {e}")
        return ""


def safe_filename(name):
    """Имя программы -> безопасное имя файла, сохраняя кириллицу."""
    name = re.sub(r"[\\/:*?\"<>|]+", " ", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name or "program"


def scrape_program(name, landing_url):
    """Собрать все страницы программы в один текст."""
    print(f"-> {name}: {landing_url}")
    landing_soup = get_soup(landing_url)
    if not landing_soup:
        return None

    pages = [landing_url]
    final = find_final_page(landing_url, landing_soup)
    if final and clean_url(final) != clean_url(landing_url):
        print(f"   конечная страница: {final}")
        pages.append(final)

    blocks = []
    file_prefix = safe_filename(name).replace(" ", "_")
    for page_url in pages:
        soup = landing_soup if page_url == landing_url else get_soup(page_url)
        if not soup:
            continue
        title, text, files = extract_post(soup)
        if not text:
            continue

        block = [f"ИСТОЧНИК: {page_url}"]
        if title:
            block.append(f"ЗАГОЛОВОК: {title}")
        block.append("")
        block.append(text)

        for fname, furl in files:
            block.append("")
            block.append(f"--- ПРИЛОЖЕННЫЙ ФАЙЛ: {fname}")
            block.append(f"ИСТОЧНИК ФАЙЛА: {furl}")
            pdf_text = download_and_read_pdf(furl, FILES_DIR / file_prefix, file_prefix)
            if pdf_text:
                block.append("ТЕКСТ ФАЙЛА:")
                block.append(pdf_text)
        blocks.append("\n".join(block))
        time.sleep(0.5)  # вежливая пауза

    if not blocks:
        print("   [!] контент не найден, пропуск")
        return None
    return ("\n\n" + "=" * 70 + "\n\n").join(blocks)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    programs = list_programs()
    print(f"Найдено программ: {len(programs)}\n")

    for name, url in programs:
        content = scrape_program(name, url)
        if not content:
            continue
        out = DATA_DIR / f"{safe_filename(name)}.txt"
        out.write_text(content, encoding="utf-8")
        print(f"   сохранено: {out.relative_to(ROOT)}  ({len(content)} символов)\n")

    print("Готово.")


if __name__ == "__main__":
    main()
