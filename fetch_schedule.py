#!/usr/bin/env python3
"""Забирает листы «Изменение на ДД.ММ.ГГ» с sch78kras.ru, находит колонку 9Б
и пишет schedule.json для виджета KWGT. Листы на сайте — PNG, поэтому OCR."""
import json, os, re, sys, datetime as dt
import cv2, numpy as np, pytesseract, requests

SITE = "https://sch78kras.ru"
CLASS_RE = re.compile(os.environ.get("CLASS_RE", r"^9\s*[Бб6]$"))
LANG = os.environ.get("OCR_LANG", "rus+eng")
OUT = os.environ.get("OUT", "schedule.json")
DEBUG = os.environ.get("DEBUG") == "1"
TITLE_RE = re.compile(r"Изменение\s+на\s+(\d{2})\.(\d{2})\.(\d{2,4})", re.I)
ROOM_RE = re.compile(r"\d-\d{2}")

# Проверено по странице «Расписание звонков» (0 урок — только понедельник)
BELLS = {0: ("08:00", "08:30"), 1: ("08:30", "09:10"), 2: ("09:20", "10:00"),
         3: ("10:15", "10:55"), 4: ("11:05", "11:45"), 5: ("11:55", "12:35"),
         6: ("12:45", "13:25"), 7: ("13:40", "14:20"), 8: ("14:30", "15:10"),
         9: ("15:15", "15:55"), 10: ("16:00", "16:40"), 11: ("16:45", "17:25"),
         12: ("17:30", "18:10")}


def log(*a):
    print(*a, file=sys.stderr)


def _groups(idx):
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= 3:
            out[-1].append(i)
        else:
            out.append([i])
    return [int(np.mean(o)) for o in out]


def grid(gray):
    """Координаты вертикальных и горизонтальных линий таблицы."""
    h, w = gray.shape
    b = (gray < 128).astype(np.uint8)
    v = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4)))
    z = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 6, 1)))
    return _groups(np.where(v.sum(axis=0) > 0)[0]), _groups(np.where(z.sum(axis=1) > 0)[0])


def up(img, k=4):
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)


def find_column(gray, xs, ys):
    for c in range(len(xs) - 1):
        cell = gray[ys[0] + 4:ys[1] - 4, xs[c] + 4:xs[c + 1] - 4]
        t = pytesseract.image_to_string(up(cell, 2), lang=LANG, config="--psm 7").strip()
        t = t.replace("\n", " ")
        if DEBUG:
            log("  header", c, repr(t))
        if CLASS_RE.match(t):
            return c
    return None


def read_cell(gray, x0, x1, y0, y1):
    cell = gray[y0 + 3:y1 - 3, x0 + 3:x1 - 3]
    if (cell < 128).mean() < 0.01:          # пустая клетка
        return None
    big = up(cell)
    H = big.shape[0]
    top = pytesseract.image_to_string(
        big[:int(H * .42)], lang="eng",
        config="--psm 7 -c tessedit_char_whitelist=0123456789-.").strip()
    m = ROOM_RE.search(top)
    room = m.group(0) if m else None
    if not room:  # «каб.12», «Спорт зал (Д)» и т.п. — берём текст как есть
        room = pytesseract.image_to_string(big[:int(H * .42)], lang=LANG, config="--psm 7").strip() or None
    mid = big[int(H * .35):int(H * .75)]
    subj = pytesseract.image_to_string(mid, lang=LANG, config="--psm 7").strip()
    subj = ROOM_RE.sub("", subj).strip(" |.\n")
    return {"subject": subj, "room": room}


def process_image(img_bytes, force_col=None):
    gray = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    xs, ys = grid(gray)
    if len(xs) < 3 or len(ys) < 3:
        return None
    c = force_col if force_col is not None else find_column(gray, xs, ys)
    if c is None:
        return None
    lessons = []
    for r in range(1, len(ys) - 1):         # строка r = урок r (0-й урок на листе не показан)
        res = read_cell(gray, xs[c], xs[c + 1], ys[r], ys[r + 1])
        if res:
            lessons.append({"n": r, **res, "start": BELLS[r][0], "end": BELLS[r][1]})
    return lessons


def get_posts():
    r = requests.get(f"{SITE}/wp-json/wp/v2/posts",
                     params={"per_page": 10, "_fields": "link,title,content"}, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    old = {}
    if os.path.exists(OUT):
        old = json.load(open(OUT, encoding="utf-8"))
    days = old.get("days", {})
    seen = set(old.get("seen", []))
    for p in get_posts():
        title = re.sub(r"<[^>]+>", "", p["title"]["rendered"])
        m = TITLE_RE.search(title)
        if not m:
            continue
        d, mo, y = m.groups()
        y = "20" + y if len(y) == 2 else y
        key = f"{y}-{mo}-{d}"
        if key in days and p["link"] in seen:
            continue
        found = None
        for src in re.findall(r'<img[^>]+src="([^"]+)"', p["content"]["rendered"]):
            src = re.sub(r"-\d+x\d+(\.\w+)$", r"\1", src)   # полный размер вместо превью
            data = requests.get(src, timeout=60).content
            found = process_image(data)
            if found:
                log("9Б найден:", key, src)
                break
        if found is not None:
            days[key] = {"source": p["link"], "lessons": found}
            seen.add(p["link"])
        else:
            log("9Б НЕ найден в", p["link"])
    cutoff = (dt.date.today() - dt.timedelta(days=10)).isoformat()
    days = {k: v for k, v in days.items() if k >= cutoff}
    json.dump({"class": "9Б", "updated": dt.datetime.now(dt.timezone.utc).isoformat(),
               "bells": {str(k): v for k, v in BELLS.items()},
               "days": dict(sorted(days.items())), "seen": sorted(seen)[-30:]},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

