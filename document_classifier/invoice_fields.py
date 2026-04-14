"""
Извлечение реквизитов из текста счёта / УПД / накладной (эвристики + регулярные выражения).
Качество зависит от вёрстки PDF/Excel; сканы без OCR не поддерживаются.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# Порядок полей для UI и целевого JSON для нейросети
FIELD_LABELS: List[str] = [
    "Банк получателя",
    "Получатель",
    "ИНН поставщика",
    "КПП поставщика",
    "ИНН покупателя",
    "КПП покупателя",
    "БИК",
    "Счет",
    "Поставщик",
    "Покупатель",
    "Номер счета",
    "Дата счета",
    "Итого",
    "Товары",
    "Количество",
    "Ед.Из",
    "Цена",
    "Сумма",
]

# Только эти ключи отдаются в API / predict после извлечения (остальное скрыто).
PUBLIC_FIELD_KEYS: Tuple[str, ...] = (
    "ИНН Поставщика",
    "Наименование поставщика",
    "СуммаИтого",
    "НомерСчета",
    "ДатаСчета",
)


def _norm(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.replace("\r\n", "\n").replace("\r", "\n")).strip()


# Родительный падеж («13 апреля») + часть именительного для дат в тексте
_RU_MONTH_TO_MM: Dict[str, str] = {
    "января": "01",
    "февраля": "02",
    "марта": "03",
    "апреля": "04",
    "мая": "05",
    "июня": "06",
    "июля": "07",
    "августа": "08",
    "сентября": "09",
    "октября": "10",
    "ноября": "11",
    "декабря": "12",
    "январь": "01",
    "февраль": "02",
    "март": "03",
    "апрель": "04",
    "май": "05",
    "июнь": "06",
    "июль": "07",
    "август": "08",
    "сентябрь": "09",
    "октябрь": "10",
    "ноябрь": "11",
    "декабрь": "12",
}


def normalize_date_display_to_ddmmyyyy(s: str) -> str:
    """
    «13 апреля 2026» → «13.04.2026»; «2026-04-13» → «13.04.2026»; уже числовые даты — с ведущими нулями.
    Неизвестный формат возвращается как есть (после _norm).
    """
    s = _norm(str(s or "").strip())
    if not s:
        return ""
    s = re.sub(r"\s*г\.?\s*$", "", s, flags=re.I).strip()
    m_iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m_iso:
        y, mo, d = m_iso.group(1), m_iso.group(2), m_iso.group(3)
        return f"{int(d):02d}.{int(mo):02d}.{y}"
    m = re.match(r"^(\d{1,2})[./](\d{1,2})[./](\d{2,4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(y) == 2:
            yi = int(y)
            y = f"20{yi:02d}" if yi < 70 else f"19{yi:02d}"
        else:
            y = str(y)
        return f"{d:02d}.{mo:02d}.{y}"
    m2 = re.match(r"^(\d{1,2})\s+([а-яё]+)\s+(\d{4})$", s, re.I)
    if m2:
        d, mon_w, y = int(m2.group(1)), m2.group(2).lower(), m2.group(3)
        mm = _RU_MONTH_TO_MM.get(mon_w)
        if mm:
            return f"{d:02d}.{mm}.{y}"
    return s


def normalize_invoice_number_ocr(s: str) -> str:
    """
    Правки типичных путаниц OCR в цифровой части номера (от первой цифры до конца).
    Префикс вроде «СКЗ» не меняем — только хвост из цифр/похожих символов.
    """
    s = (s or "").strip()
    if not s:
        return ""
    cut = 0
    for i, c in enumerate(s):
        if c.isdigit():
            cut = i
            break
    else:
        return s[:120]
    head, tail = s[:cut], s[cut:]
    trans = str.maketrans(
        {
            "О": "0",
            "о": "0",
            "O": "0",
            "З": "3",
            "з": "3",
            "l": "1",
            "|": "1",
            "I": "1",
        },
    )
    return (head + tail.translate(trans))[:120]


def _strip_supplier_name_suffix(s: str) -> str:
    """
    Убирает слепленный к ФИО/ООО хвост: телефон, банк, БИК, город банка, счёт
    (часто всё в одной строке без перевода строки).
    """
    s = _norm(str(s or "").strip())
    if not s:
        return ""
    cut_at: List[int] = []
    for pat in (
        r",\s*тел\.",
        r"\s+тел\.\s*[\d\+\-\(\)\s]{7,22}",
        r"\s*\(\s*\d{1,3}\s*\)\s*Адрес\b",
        r"\s+Адрес\s*\d{0,6}\b",
        r"\s+ИНН/КПП\b",
        r"\s+Получатель\s*[:：]",
        r"\s+Покупатель\s*[:：]",
        r"\s+Заказчик\s*[:：]",
        r"\s+Отправитель\s*[:：]",
        r"\s+СЕВЕРО[-\s]?ЗАПАДНЫЙ\s+БАНК",
        r"\s+БАНК\s+ПАО",
        r"\s+ПАО\s+СБЕРБАНК",
        r"\s+АО\s+БАНК",
        r"\s+БИК\s*\d{8,9}",
        r"\s+г\.?\s+[Сс]анкт[-\s]?[Пп]етербург",
        r"\s+Сч\.?\s*№",
        r"(?i)банк\s+получателя",
    ):
        m = re.search(pat, s)
        if m:
            cut_at.append(m.start())
    if cut_at:
        s = s[: min(cut_at)].strip()
    return re.sub(r",\s*$", "", s).strip()


def supplier_display_name(supplier_line: str) -> str:
    """
    Из строки поставщика (часто «ООО …, 192249, г., ул. …») оставляет наименование без индекса/адреса.
    """
    s = _trim_party_block(supplier_line or "")
    s = _strip_supplier_name_suffix(s)
    if not s:
        return ""
    m = re.match(r"^(.+?),\s*\d{6}\b", s)
    if m and len(m.group(1).strip()) > 3:
        return _norm(m.group(1).strip())[:500]
    m = re.match(r"^(.+?),\s*(?:[гГ]\.|город|[Сс]анкт|[Мм]осква)", s)
    if m and len(m.group(1).strip()) > 5:
        return _norm(m.group(1).strip())[:500]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) >= 2:
        tail_join = ",".join(parts[1:])
        if re.search(r"(?:ул\.?|улица|просп\.|переул|д\.|к\.|кв\.|стр\.|пом\.)", tail_join, re.I):
            return _norm(parts[0])[:500]
        if re.match(r"^\d{6}$", parts[1].replace(" ", "")):
            return _norm(parts[0])[:500]
    return s[:500]


def project_public_fields(internal: Dict[str, Any]) -> Dict[str, str]:
    """Сужает полный словарь извлечения до PUBLIC_FIELD_KEYS для ответа API."""
    inn = str(internal.get("ИНН поставщика") or "").strip()
    supplier_line = str(internal.get("Поставщик") or "").strip()
    name = supplier_display_name(supplier_line)
    total = str(internal.get("Итого") or "").strip()
    inv_no = normalize_invoice_number_ocr(str(internal.get("Номер счета") or "").strip())
    inv_dt = normalize_date_display_to_ddmmyyyy(str(internal.get("Дата счета") or "").strip())
    return {
        "ИНН Поставщика": inn,
        "Наименование поставщика": name,
        "СуммаИтого": total,
        "НомерСчета": inv_no,
        "ДатаСчета": inv_dt,
    }


def _is_likely_bank_rs_number(s: str) -> bool:
    """Расчётный/корр. счёт (часто 20 цифр), не номер счёта на оплату."""
    d = re.sub(r"\D", "", str(s or ""))
    return bool(d) and d.isdigit() and 18 <= len(d) <= 22


def _extract_invoice_number(text: str) -> str:
    """Номер счёта в шапке: «Счёт на оплату № 13», «СЧЕТ НА ЗАЛОГ … № 5717 от»."""
    head = text[:4000] if text else ""
    # Сначала явный заголовок документа — не путать с «Сч. №» банковского счёта (20 цифр)
    m = re.search(
        r"(?:Сч[её]т|СЧЕТ)\s+на\s+оплату\s*(?:№|N[oо]\.?)\s*([\dA-Za-zА-Яа-яЁё0-9][\dA-Za-zА-Яа-яЁё\-/]*)",
        head,
        re.I,
    )
    if m:
        s = _norm(m.group(1)).strip()
        s = re.split(r"\s+от\s+", s, maxsplit=1, flags=re.I)[0].strip()
        if s and not _is_likely_bank_rs_number(s):
            return normalize_invoice_number_ocr(s[:120])
    # Между «Счёт» и «№» может быть длинная фраза (залог, тара и т.д.)
    m = re.search(
        r"(?:Сч[её]т|СЧЕТ)\s+[^\n]{0,400}?(?:№|N[oо]\.?)\s*([\dA-Za-zА-Яа-яЁё][\dA-Za-zА-Яа-яЁё\-/]*)(?:\s+от\b|\s*\n|$)",
        head,
        re.I,
    )
    if m:
        s = _norm(m.group(1)).strip()
        s = re.split(r"\s+от\s+", s, maxsplit=1, flags=re.I)[0].strip()
        if not _is_likely_bank_rs_number(s):
            return normalize_invoice_number_ocr(s[:120])
    m = re.search(
        r"(?:Сч[её]т|СЧЕТ)(?:\s+на\s+оплату)?\s*(?:№|N[oо]\.?)\s*([^\n]+?)(?:\s+от\s+|\n|$)",
        head,
        re.I,
    )
    if m:
        s = _norm(m.group(1))
        s = re.split(r"\s+от\s+", s, maxsplit=1, flags=re.I)[0].strip()
        if not _is_likely_bank_rs_number(s):
            return normalize_invoice_number_ocr(s[:120])
    m = re.search(r"(?:^|\n)\s*№\s*(\d[\d\w\-/]*)\s+от\s+", head, re.I)
    if m:
        cand = m.group(1).strip()
        if not _is_likely_bank_rs_number(cand):
            return normalize_invoice_number_ocr(cand[:120])
    # Буквенно-цифровой номер (напр. СКЗ00021097) рядом со «Счёт … №»
    m = re.search(
        r"(?:Сч[её]т|СЧЕТ)[^\n]{0,200}?(?:№|N[oо]\.?)\s*([A-Za-zА-Яа-яЁё]{1,12}[\dA-Za-zА-Яа-яЁё\-/]*)",
        head,
        re.I,
    )
    if m:
        s = _norm(m.group(1)).strip()
        s = re.split(r"\s+", s)[0] if s else ""
        return normalize_invoice_number_ocr(s[:120])
    m = re.search(
        r"(?:^|\n)\s*№\s*([A-Za-zА-Яа-яЁё]{1,12}[\dA-Za-zА-Яа-яЁё\-/]*)(?:\s+от\s+|\s*\n|$)",
        head,
        re.I,
    )
    if m:
        s = _norm(m.group(1)).strip().split()[0]
        return normalize_invoice_number_ocr(s[:120])
    return ""


def _extract_invoice_date(text: str) -> str:
    """Дата рядом со счётом: от 28.03.2026, от 28 марта 2026 г."""
    head = text[:12000] if text else ""
    # «СЧЕТ … № 5717 от 13 апреля 2026» — длинный текст между «Счёт» и «№»
    m = re.search(
        r"(?:Сч[её]т|СЧЕТ)[^\n]{0,400}?(?:№|N[oо]\.?)\s*[\dA-Za-zА-Яа-яЁё\-/]+\s+от\s+(\d{1,2}\s+[а-яё]+\s+\d{4})\s*г?\.?",
        head,
        re.I,
    )
    if m:
        return _norm(m.group(1))[:80]
    # Длинная строка «Счет №26-… от 19.03.2026 г.» — до 200 символов между «Счет» и «от»
    m = re.search(
        r"(?:Сч[её]т|СЧЕТ)[^\n]{0,220}?\s+от\s+(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})",
        head,
        re.I,
    )
    if m:
        return m.group(1).strip()
    # Номер и дата в одной фразе без «Счет» в начале строки
    m = re.search(
        r"(?:№|N[oо]\.?)\s*[^\n]{0,120}?\s+от\s+(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})",
        head,
        re.I,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r"\s+от\s+(\d{1,2}\s+[а-яё]+\s+\d{4})\s*г?",
        head,
        re.I,
    )
    if m:
        return _norm(m.group(1))[:80]
    return ""


def _extract_itogo_total(text: str) -> str:
    """Итоговая сумма к оплате (не сумма строки таблицы)."""
    t = text or ""
    # Если есть итог "с НДС", он приоритетнее обычного "Итого"
    for pat in (
        r"(?:Итого|Всего)\s+с\s+НДС\s*[:\s]*([\d\s\u00a0]+(?:[,.]\d{2})?)",
        r"(?:Сумма|Итого)\s+с\s+НДС\s*[:\s]*([\d\s\u00a0]+(?:[,.]\d{2})?)",
    ):
        m = re.search(pat, t, re.I)
        if m:
            s = m.group(1).replace("\xa0", " ").strip()
            s = re.sub(r"\s+", "", s)
            return s[:32]

    for pat in (
        # «ИТОГО К ОПЛАТЕ: 8 858,00» — раньше ловилось только «Итого» и ломалось на «К ОПЛАТЕ»
        r"(?:Итого\s+к\s+оплате)\s*[:\s]*([\d\s\u00a0]+(?:[,.]\d{2})?)\s*(?:руб|₽)?",
        r"(?:Итого|Всего\s+к\s+оплате|К\s+оплате)\s*[:\s]*([\d\s\u00a0]+(?:[,.]\d{2})?)\s*(?:руб|₽)?",
        r"(?:Всего)\s+(?:наименований\s+\d+\s+)?на\s+сумму\s+([\d\s\u00a0]+(?:[,.]\d{2})?)",
        r"(?:Сумма|Итого)\s+по\s+сч[её]ту\s*[:\s]*([\d\s\u00a0]+(?:[,.]\d{2})?)",
        r"(?:К\s+оплате|Оплатить)\s*[:\s]*([\d\s\u00a0]+(?:[,.]\d{2})?)\s*(?:руб|₽)?",
    ):
        m = re.search(pat, t, re.I)
        if m:
            s = m.group(1).replace("\xa0", " ").strip()
            s = re.sub(r"\s+", "", s)
            return s[:32]
    return ""


def is_itogo_amount_in_words(s: str) -> bool:
    """
    Сумма прописью («Семь тысяч … рублей … копеек»), а не число из строки «Итого к оплате».
    Нужно, чтобы merge не предпочитал такой ответ LLM числовому regex.
    """
    t = _norm(str(s or "").strip())
    if len(t) < 16:
        return False
    low = t.lower()
    # Уже цифры + копейки (как в документе)
    if re.search(r"\d\s*[\d\s\u00a0]*[,.]\s*\d{1,2}", t) and len(t) < 45:
        return False
    if re.match(r"^[\d\s\u00a0,.]+(?:руб|₽)?\.?\s*$", t, re.I):
        return False
    if "рубл" not in low and "копе" not in low:
        return False
    letters = len(re.findall(r"[а-яёА-ЯЁ]", t))
    return letters >= 10


def _first_match(patterns: List[str], text: str, flags: int = re.IGNORECASE | re.MULTILINE) -> str:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return _norm(m.group(1) if m.lastindex else m.group(0))
    return ""


def _all_inns(text: str) -> List[str]:
    found: List[str] = []
    seen = set()
    for m in re.finditer(
        r"(?:ИНН|Инн)\s*[\/]?\s*(?:КПП\s*)?[:\s]*(\d{10}|\d{12})(?!\d)",
        text,
        re.IGNORECASE,
    ):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            found.append(m.group(1))
    return found


def _kpps(text: str) -> List[str]:
    out = []
    for m in re.finditer(
        r"(?:КПП|кпп)\s*[:\s\/]*(\d{9})(?!\d)",
        text,
        re.IGNORECASE,
    ):
        out.append(m.group(1))
    # ИНН/КПП одной строкой
    for m in re.finditer(
        r"(\d{10}|\d{12})\s*\/\s*(\d{9})",
        text,
    ):
        out.append(m.group(2))
    return list(dict.fromkeys(out))


def _biks(text: str) -> List[str]:
    found = []
    for m in re.finditer(r"(?:БИК|бик)\s*[:\s]*(\d{9})(?!\d)", text, re.IGNORECASE):
        found.append(m.group(1))
    return list(dict.fromkeys(found))


def _accounts_20(text: str) -> List[str]:
    """Расчётный / корр. счёт — 20 цифр. Сначала р/с, не корр. счёт."""
    rs: list[str] = []
    ks: list[str] = []
    for m in re.finditer(
        r"(?:р[/\\]?с|р\.?\s*счет|расчетн\w*\s+счет)\s*[:\s№]*(\d{20})",
        text,
        re.IGNORECASE,
    ):
        rs.append(m.group(1))
    for m in re.finditer(
        r"(?:к[/\\]?с|корр\.?\s*счет)\s*[:\s№]*(\d{20})",
        text,
        re.IGNORECASE,
    ):
        ks.append(m.group(1))
    if rs:
        return list(dict.fromkeys(rs))
    if ks:
        return list(dict.fromkeys(ks))
    found = []
    for m in re.finditer(
        r"(?:счет\s*№|счёт\s*№)\s*[:\s]*(\d{20})",
        text,
        re.IGNORECASE,
    ):
        found.append(m.group(1))
    if not found:
        for m in re.finditer(r"(?<![\d])(\d{20})(?![\d])", text):
            found.append(m.group(1))
    return list(dict.fromkeys(found))


def _block_after_keyword(text: str, keywords: Tuple[str, ...], stop_keywords: Tuple[str, ...]) -> str:
    t = text
    low = t.lower()
    start = -1
    for kw in keywords:
        idx = low.find(kw.lower())
        if idx != -1 and (start == -1 or idx < start):
            start = idx + len(kw)
    if start < 0:
        return ""
    rest = t[start:]
    stop_at = len(rest)
    for sk in stop_keywords:
        si = rest.lower().find(sk.lower())
        if si != -1 and 0 < si < stop_at:
            stop_at = si
    block = rest[:stop_at]
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    cleaned = []
    for ln in lines[:12]:
        if re.match(r"^(ИНН|КПП|БИК|р/с|к/с|тел|телефон|email)[\s:]", ln, re.I):
            continue
        if len(ln) > 2:
            cleaned.append(ln)
    out = _norm(" ".join(cleaned[:5]))
    out = re.sub(r"^[:\s\-–]+", "", out)
    return out


def _is_bad_bank_name(s: str) -> bool:
    """Не банк: одно слово «ИНН», строка реквизитов и т.п."""
    s = (s or "").strip()
    if len(s) < 3:
        return True
    low = s.lower()
    if low in ("инн", "кпп", "бик", "р/с", "к/с"):
        return True
    if re.match(r"^(?:бик|инн|кпп|р[/\\]?с|к[/\\]?с|корр|счет|счёт)\s*[/:]?\s*", s, re.I):
        return True
    if re.match(r"^инн\s*[/]?\s*[\d\s]*$", s, re.I):
        return True
    if re.match(r"^кпп\s*[/]?\s*[\d\s]*$", s, re.I):
        return True
    if re.fullmatch(r"[\d\s]{9,}", s.replace(" ", "")):
        return True
    if re.match(r"^Сч\.?\s*№", s, re.I):
        return True
    return False


# Не матчить «получатель» внутри «грузополучатель» и т.п.
_RECIPIENT_KW = r"(?<![а-яёА-ЯЁa-zA-Z])(?:Получатель\s+платежа|Получатель)(?![а-яёА-ЯЁa-zA-Z])"
_RECIPIENT_NAME_KW = r"(?<![а-яёА-ЯЁa-zA-Z])Наименование\s+получателя(?![а-яёА-ЯЁa-zA-Z])"


def _is_bad_recipient(s: str) -> bool:
    """Фрагменты заголовка («оплаты») и прочий мусор вместо наименования."""
    s = _norm(str(s or "").strip())
    if len(s) < 2:
        return True
    low = s.lower()
    if low in ("оплаты", "оплату", "оплата", "на оплату", "оплат"):
        return True
    if re.match(r"^(оплат|сч[её]т|счёт)\b", s, re.I):
        return True
    if re.search(r"сч[её]т\s+на\s+оплату", s, re.I):
        return True
    if re.match(r"^сч[её]т\s+№", s, re.I):
        return True
    # Кусок адреса вместо юрлица (улица/дом/корпус/помещение)
    if re.search(r"\b(ул\.?|улица|просп\.?|проспект|пер\.?|переулок|дом|д\.|корп\.?|корпус|пом\.?|помещение)\b", low, re.I):
        return True
    # Частый OCR/LLM артефакт: одиночное прилагательное из адреса («БРАТИСЛАВСКАЯ», «ЛЕНИНСКИЙ»)
    if " " not in s and len(s) >= 8 and re.match(r"^[А-ЯЁA-Z\-]+$", s) and re.search(r"(СКАЯ|СКИЙ|СКОЙ|СКОЕ)$", s):
        return True
    # Юридические формулировки из оферты/условий вместо названия контрагента
    if re.search(
        r"(обязует(?:ся|ься)\s+постав|обязует(?:ся|ься)\s+оплат|обязуется\s+принять|в\s+соответствии\s+с\s+настоящ)",
        low,
        re.I,
    ):
        return True
    # Фрагмент из таблицы/колонок (часто подставляет LLM вместо организации)
    if re.search(r"заказной\s+товар", low):
        return True
    if re.match(r"^[A-Za-zА-Яа-яЁё0-9]{1,4}\s*,\s*[A-Za-zА-Яа-яЁё0-9]{1,4}\b", s) and re.search(
        r"товар|постав",
        low,
    ):
        return True
    return False


def _first_bank_line_from_block(block: str) -> str:
    """После «Банк получателя» часто идут ИНН/КПП — пропускаем, ищем строку с названием."""
    for ln in block.split("\n"):
        cand = _norm(ln)
        if not cand:
            continue
        if _is_bad_bank_name(cand):
            continue
        if re.match(r"^(?:БИК|ИНН|КПП|р[/\\]?с|к[/\\]?с|корр|счет|счёт)\b", cand, re.I):
            continue
        if re.match(r"^р[/\\]?с\s*№?\s*[\d\s]+$", cand, re.I):
            continue
        if re.match(r"^Сч\.?\s*№", cand, re.I):
            continue
        if re.match(r'^ООО\s+[«"]', cand, re.I):
            continue
        return cand[:500]
    return ""


def _bank_name(text: str) -> str:
    # Сначала шапка: «ПАО …банк» / строка с «БИК» на следующей строке
    lines = text.replace("\r\n", "\n").split("\n")
    for i, ln in enumerate(lines[:35]):
        cand = _norm(ln)
        if len(cand) < 6:
            continue
        if _is_bad_bank_name(cand):
            continue
        if re.match(r"^(?:Сч\.|Счет|р[/\\]?с|к[/\\]?с)", cand, re.I):
            continue
        if re.search(
            r"(?:ПАО|АО|ОАО|ЗАО|НКО|банк|БАНК|инвестбанк|сбербанк|тинькофф|альфа|втб|челяб)",
            cand,
            re.I,
        ):
            nxt = _norm(lines[i + 1]) if i + 1 < len(lines) else ""
            if re.match(r"^БИК\s*$", nxt, re.I) or re.match(r"^БИК\s+", nxt, re.I):
                return cand[:500]
    patterns = [
        r"(?:Банк\s+получателя|Наименование\s+банка)\s*[:\s]*\n([\s\S]+?)(?=\n\s*(?:БИК|ИНН|р[/\\]?с|р\.?\s*счет|к[/\\]?с|корр)|\Z)",
        r"(?:Банк\s+получателя|Наименование\s+банка)\s*[:\s]+(.+?)(?:\n|$)",
        r"р[/\\]?с\s*(?:№)?\s*[\d\s]+\s+в\s+банке\s+(.+?)(?:\n|БИК)",
        r"(?:в\s+банке)\s*[:\s]+(.+?)(?:\n|БИК)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            chunk = m.group(1) if m.lastindex else m.group(0)
            cand = _first_bank_line_from_block(chunk)
            if cand:
                return cand
            cand = _norm(chunk.split("\n")[0])
            if cand and not _is_bad_bank_name(cand):
                return cand[:500]
    return ""


def _recipient_name(text: str) -> str:
    # Получатель платежа / наименование получателя
    s = _first_match(
        [
            rf"{_RECIPIENT_KW}\s*[:\s]+(.+?)(?:\n|$)",
            rf"{_RECIPIENT_NAME_KW}\s*[:\s]+(.+?)(?:\n|$)",
        ],
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if s:
        line = _norm(s.split("\n")[0][:500])
        if re.match(r"^(сч[её]т|счёт)\s+(на\s+оплату|№)", line, re.I):
            return ""
        if _is_bad_recipient(line):
            return ""
        return line
    return ""


def _recipient_before_label(text: str) -> str:
    """Строка «ООО …» сразу перед отдельной строкой «Получатель» (типичная вёрстка PDF)."""
    m = re.search(
        rf"(?:^|\n)\s*(.+?)\s*\n\s*{_RECIPIENT_KW}\s*(?:\n|$)",
        text,
        re.I | re.MULTILINE,
    )
    if m:
        cand = _norm(m.group(1))
        if len(cand) > 3 and not re.match(r"^Сч\.?\s*№", cand, re.I):
            if re.search(r"сч[её]т\s+на\s+оплату", cand, re.I):
                return ""
            if _is_bad_recipient(cand):
                return ""
            return cand[:500]
    return ""


def _recipient_safe(text: str) -> str:
    """
    Получатель денег (не путать с «Банк получателя»: там тоже есть слово «получатель»).
    """
    r = _recipient_name(text)
    if r:
        return r
    for m in re.finditer(
        rf"(?:^|\n)\s*{_RECIPIENT_KW}\s*[:\s]+(.+?)(?=\n|$)",
        text,
        re.I | re.MULTILINE,
    ):
        ctx_start = max(0, m.start() - 30)
        ctx = text[ctx_start : m.start() + 5]
        if re.search(r"банк\s+получателя", ctx, re.I):
            continue
        frag = m.group(1).strip()
        if re.match(r"^(сч[её]т|счёт)\s+(на\s+оплату|№)", frag, re.I):
            continue
        line = _norm(frag.split("\n")[0][:400])
        if _is_bad_recipient(line):
            continue
        return line
    return ""


def _line_after_party_keyword(text: str, keywords: Tuple[str, ...]) -> str:
    """Наименование после «Поставщик»/«Покупатель» до «, ИНН» (учёт строки «(Исполнитель):»)."""
    role_skip = r"(?:\(\s*(?:[Ии]сполнитель|[Зз]аказчик)\s*\)\s*:?\s*\n)?"
    for kw in keywords:
        m = re.search(
            rf"(?:^|\n)\s*{re.escape(kw)}\s*[:\s]*\n\s*{role_skip}([\s\S]+?)(?=,\s*ИНН\b|,\s*КПП\b)",
            text,
            re.IGNORECASE,
        )
        if m:
            line = _norm(m.group(1).replace("\n", " ").strip())
            line = re.sub(r"^\(\s*[Ии]сполнитель\s*\)\s*:?\s*", "", line, flags=re.I)
            line = re.sub(r"^\(\s*[Зз]аказчик\s*\)\s*:?\s*", "", line, flags=re.I)
            if len(line) > 1 and not re.match(r"^(сч[её]т|счёт)\s+(на\s+оплату|№)", line, re.I):
                return line[:500]
    return ""


def _trim_party_block(s: str, max_len: int = 400) -> str:
    """Одна-две строки, без хвоста с ИНН/КПП/адресом в одну кашу."""
    if not s:
        return ""
    s = _norm(s)
    s = re.sub(r"^\(?\s*(исполнитель|поставщик|продавец|заказчик|покупатель)\s*\)?\s*[:\s]*", "", s, flags=re.I)
    s = re.sub(r"^\)\s*:\s*", "", s)
    # Дефис только как символ, не диапазон «(—…» (иначе съедается кириллическая «О» в «ООО»)
    s = re.sub(r"^[:\s()\-–—]+", "", s)
    parts = re.split(r"\s*,\s*(?=ИНН|КПП|ОГРН|ОКПО)", s, flags=re.I)
    head = parts[0] if parts else s
    head = re.split(r"\s+ИНН\s+", head, maxsplit=1, flags=re.I)[0].strip()
    lines = [ln.strip() for ln in head.split("\n") if ln.strip()]
    short = " ".join(lines[:3]) if lines else head
    return short[:max_len].strip()


def _entity_name_near_inn(text: str, inn: str) -> str:
    """
    Наименование из «Получатель: …» / «Отправитель: …» / «Покупатель: …», если в следующих строках указан этот ИНН
    (типично для транспортных/сервисных счетов без слова «Поставщик»).
    """
    if not inn or not text:
        return ""
    t = text
    for m in re.finditer(
        r"(?:Получатель|Отправитель|Поставщик|Исполнитель|Продавец|Покупатель|Заказчик)\s*[:\s]+(.+?)(?=\n\s*(?:ИНН|ИНН/КПП|Инн|КПП|БИК|р[/\\]?с)|\n\n|\Z)",
        t,
        re.I | re.DOTALL,
    ):
        rest = t[m.end() : m.end() + 600]
        if inn in rest:
            line = _norm(m.group(1).replace("\n", " ").strip())
            line = line.split(";")[0].strip()
            if len(line) > 2 and not _is_bad_recipient(line):
                return _trim_party_block(line)[:500]
    for m in re.finditer(
        rf"(?:ИНН|Инн)\s*[/]?\s*(?:КПП\s*)?[:\s]*{re.escape(inn)}(?:\D|$)",
        t,
        re.I,
    ):
        prev = t[max(0, m.start() - 900) : m.start()]
        for pm in re.finditer(
            r"(?:Получатель|Отправитель|Поставщик|Исполнитель|Покупатель|Заказчик)\s*[:\s]+(.+?)(?:\n|$)",
            prev,
            re.I,
        ):
            line = _norm(pm.group(1).split("\n")[0])
            if len(line) > 2 and not _is_bad_recipient(line):
                return _trim_party_block(line)[:500]
    return ""


def _supplier_buyer(text: str) -> Tuple[str, str]:
    sup = _line_after_party_keyword(
        text,
        ("поставщик", "продавец", "исполнитель"),
    )
    if not sup:
        sup = _block_after_keyword(
            text,
            ("исполнитель", "поставщик", "продавец"),
            (
                "заказчик",
                "покупатель",
                "плательщик",
                "грузополучатель",
                "банк получателя",
                "банковские реквизиты",
            ),
        )
        sup = _trim_party_block(sup)
    else:
        sup = _trim_party_block(sup)

    buy = _line_after_party_keyword(
        text,
        ("покупатель", "заказчик", "плательщик", "грузополучатель"),
    )
    if not buy:
        buy = _block_after_keyword(
            text,
            ("заказчик", "покупатель", "плательщик"),
            (
                "банк получателя",
                "банковские реквизиты",
                "реквизиты банка",
                "бик",
                "р/с",
                "р/счет",
                "расчетный счет",
                "расчётный счет",
                "к/с",
                "итого",
                "всего к оплате",
                "наименование",
                "номенклатура",
                "основание",
            ),
        )
        buy = _trim_party_block(buy)
    else:
        buy = _trim_party_block(buy)

    return sup, buy


def _parse_table_rows_stacked(lines: List[str], header_idx: int) -> List[Dict[str, str]]:
    """
    Вёрстка как в многих PDF: № строки, наименование (несколько строк), «11 м3», цена, сумма.
    """
    rows: List[Dict[str, str]] = []
    # «м3» в PDF часто латинская m + цифра 3
    qty_unit_re = re.compile(
        r"^(\d+(?:[.,]\d+)?)\s+([мm]3|[мm]\^3|[мm]\^?3|[ГG]кал|гкал|шт|м2|ч|усл|компл|ед)\s*$",
        re.I,
    )
    n = len(lines)
    i = header_idx + 1 if header_idx >= 0 else 0
    while i < n:
        ln = lines[i].strip()
        if re.match(r"^\d{1,3}$", ln) and len(ln) <= 3 and ln not in ("№", "0"):
            nxt = lines[i + 1].strip() if i + 1 < n else ""
            if len(nxt) > 2 and not re.match(r"^(Кол|Ед|Цена|Сумма|Товары|№)\s*$", nxt, re.I):
                break
        i += 1
    if i >= n:
        i = header_idx + 1 if header_idx >= 0 else 0
    while i < n:
        raw = lines[i].strip()
        if re.match(r"^(итого|всего к оплате|без налога|всего наименований)", raw, re.I):
            break
        if re.match(r"^\d{1,3}$", raw) and len(raw) <= 3:
            i += 1
            name_parts: List[str] = []
            while i < n:
                line = lines[i].strip()
                if re.match(r"^(итого|всего|без налога)", line, re.I):
                    return rows
                if re.match(r"^\d{1,3}$", line) and len(line) <= 3 and name_parts:
                    break
                mu = qty_unit_re.match(line)
                if mu:
                    qty, unit = mu.group(1), mu.group(2)
                    i += 1
                    price = lines[i].strip() if i < n else ""
                    i += 1
                    sumv = lines[i].strip() if i < n else ""
                    i += 1
                    name = _norm(" ".join(name_parts))[:2000]
                    if name or qty:
                        rows.append(
                            {
                                "name": name,
                                "qty": qty.replace(",", "."),
                                "unit": unit,
                                "price": price,
                                "sum": sumv,
                            },
                        )
                    break
                name_parts.append(line)
                i += 1
            continue
        i += 1
    return rows


def _parse_table_rows(text: str) -> List[Dict[str, str]]:
    """
    Ищет строки табличной части (номенклатура, кол-во, ед., цена, сумма).
    Возвращает список словарей с ключами: name, qty, unit, price, sum
    """
    lines = [_norm(ln) for ln in text.split("\n")]
    header_idx = -1
    # Сначала явная строка таблицы — иначе «Всего наименований … на сумму» ловится как заголовок
    for i, ln in enumerate(lines):
        if re.search(r"Товары\s*\(работы", ln, re.I):
            header_idx = i
            break
    if header_idx < 0:
        for i, ln in enumerate(lines):
            if re.search(r"номенклатур|наименовани|товар|работ|услуг", ln, re.I) and re.search(
                r"кол|цен|сумм|ед\.?\s*изм",
                ln,
                re.I,
            ):
                if re.search(r"всего\s+наименований|итого\s*:|всего\s+к\s+оплате", ln, re.I):
                    continue
                header_idx = i
                break
    if header_idx < 0:
        for i, ln in enumerate(lines):
            if re.search(r"цена", ln, re.I) and re.search(r"сумма", ln, re.I):
                header_idx = i
                break
    if header_idx < 0:
        for i, ln in enumerate(lines):
            if re.search(r"кол-?\s*во|количество", ln, re.I) and re.search(
                r"цен|сумм|ед",
                ln,
                re.I,
            ):
                header_idx = i
                break
    if header_idx < 0:
        for i, ln in enumerate(lines):
            if ln.count("\t") >= 4:
                header_idx = i - 1 if i > 0 else i
                break

    rows: List[Dict[str, str]] = []
    if header_idx < 0:
        return rows

    for ln in lines[header_idx + 1 :]:
        if not ln.strip():
            continue
        low = ln.lower()
        if re.search(r"^(итого|всего|сумма\s*ндс|в\s*том\s*числе)", low):
            break
        if re.match(r"^\d+\s*$", ln):
            continue

        parts = ln.split("\t") if "\t" in ln else re.split(r"\s{2,}", ln)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            continue

        # эвристика: последние числовые поля — сумма, цена, количество
        nums: List[str] = []
        name_parts: List[str] = []
        for p in parts:
            if re.match(r"^[\d\s]+[.,]?\d*$", p.replace(" ", "")) or re.match(
                r"^\d+[.,]\d{2}$",
                p.replace(" ", ""),
            ):
                nums.append(p.replace(" ", ""))
            else:
                name_parts.append(p)

        name = _norm(" ".join(name_parts)) if name_parts else ""
        if not name and parts:
            name = parts[0]

        row = {"name": name, "qty": "", "unit": "", "price": "", "sum": ""}
        if len(nums) >= 1:
            row["sum"] = nums[-1]
        if len(nums) >= 2:
            row["price"] = nums[-2]
        if len(nums) >= 3:
            row["qty"] = nums[-3]
        # ед. изм. — короткое слово (шт, кг, ч, м)
        for p in parts:
            if re.match(
                r"^(шт|кг|т|м|м2|м3|м\^?3|ч|час|усл|упак|компл|ед|л|гкал|гкал\.?)\.?$",
                p,
                re.I,
            ):
                row["unit"] = p
                break

        if name or any(row[k] for k in ("qty", "price", "sum")):
            rows.append(row)

    if not rows and header_idx >= 0:
        rows = _parse_table_rows_stacked(lines, header_idx)

    if not rows:
        for ln in lines:
            m = re.match(
                r"^(.+?)\s+(\d+(?:[.,]\d+)?)\s+(шт|кг|м2?|м3|м\^?3|ч|усл|компл|ед|т|упак|гкал)\s+(\d+[.,]\d+)\s+(\d+[.,]\d+)\s*$",
                ln,
                re.I,
            )
            if m:
                rows.append(
                    {
                        "name": _norm(m.group(1)),
                        "qty": m.group(2).replace(",", "."),
                        "unit": m.group(3),
                        "price": m.group(4).replace(",", "."),
                        "sum": m.group(5).replace(",", "."),
                    },
                )

    # Строки с «м3», «Гкал» и числами в конце (вёрстка PDF с пробелами)
    if not rows:
        for ln in lines:
            if not re.search(r"(м3|м\^3|гкал|Гкал)", ln, re.I):
                continue
            num_chunks = re.findall(r"[\d\s]+[.,]?\d*", ln)
            num_chunks = [n.replace(" ", "").replace(",", ".") for n in num_chunks if re.search(r"\d", n)]
            if len(num_chunks) >= 3:
                unit_m = re.search(r"(м3|м\^3|гкал|Гкал|шт)", ln, re.I)
                unit = unit_m.group(1) if unit_m else ""
                name = re.split(r"\s+\d", ln, maxsplit=1)[0].strip()
                name = _norm(" ".join(name.split())[:500])
                if len(name) > 3:
                    rows.append(
                        {
                            "name": name,
                            "qty": num_chunks[0] if num_chunks else "",
                            "unit": unit,
                            "price": num_chunks[-2] if len(num_chunks) >= 2 else "",
                            "sum": num_chunks[-1] if num_chunks else "",
                        },
                    )

    return rows[:50]


def joined_fields_from_items(items: List[Dict[str, Any]]) -> Dict[str, str]:
    """Строки Товары/Количество/… как в табличной части (через « | »)."""
    if not items:
        return {}
    names: List[str] = []
    qtys: List[str] = []
    units: List[str] = []
    prices: List[str] = []
    sums: List[str] = []
    for r in items:
        if not isinstance(r, dict):
            continue
        names.append(str(r.get("name", "")))
        qtys.append(str(r.get("qty", "")))
        units.append(str(r.get("unit", "")))
        prices.append(str(r.get("price", "")))
        sums.append(str(r.get("sum", "")))
    return {
        "Товары": " | ".join(names)[:2000],
        "Количество": " | ".join(qtys),
        "Ед.Из": " | ".join(units),
        "Цена": " | ".join(prices),
        "Сумма": " | ".join(sums),
    }


def extract_invoice_fields(text: str) -> Dict[str, Any]:
    """
    Возвращает словарь с русскими ключами из FIELD_LABELS + 'items' (список позиций).
    """
    text = text or ""
    if not text.strip():
        return {k: "" for k in FIELD_LABELS} | {"items": []}

    t = _norm(text)
    full = text  # для многострочного поиска блоков

    bank = _bank_name(full)
    if not bank:
        fb = _first_match(
            [r"(?:в\s+банке)\s*[:\s]+(.+?)(?:\n|БИК)"],
            full,
            re.I | re.DOTALL,
        )
        if not _is_bad_bank_name(fb):
            bank = fb
    if _is_bad_bank_name(bank):
        bank = ""

    recipient = _recipient_safe(full)
    if not recipient.strip():
        recipient = _recipient_before_label(full)
    if _is_bad_recipient(recipient):
        recipient = ""

    inns = _all_inns(full)
    kpps = _kpps(full)

    biks = _biks(full)
    bik_str = biks[0] if biks else ""

    accs = _accounts_20(full)
    acc_str = accs[0] if accs else ""

    supplier, buyer = _supplier_buyer(full)

    if not recipient.strip() and supplier.strip():
        recipient = supplier

    inv_num = _extract_invoice_number(full)
    inv_date = _extract_invoice_date(full)
    itogo = _extract_itogo_total(full)

    items = _parse_table_rows(full)
    if items:
        jf = joined_fields_from_items(items)
        names_join = jf["Товары"]
        qty_join = jf["Количество"]
        unit_join = jf["Ед.Из"]
        price_join = jf["Цена"]
        sum_join = jf["Сумма"]
    else:
        names_join = _first_match(
            [r"(?:основание|назначение\s+платежа|за\s+что)\s*[:\s]+(.+?)(?:\n|$)"],
            full,
            re.I | re.DOTALL,
        )
        qty_join = unit_join = price_join = sum_join = ""

    inn_sup = inn_buy = ""
    if len(inns) >= 2:
        inn_sup, inn_buy = inns[0], inns[1]
    elif len(inns) == 1:
        inn_sup = inns[0]

    kpp_sup = kpp_buy = ""
    if len(kpps) >= 2:
        kpp_sup, kpp_buy = kpps[0], kpps[1]
    elif len(kpps) == 1:
        kpp_sup = kpps[0]

    if _is_bad_recipient(supplier):
        supplier = ""
    if not supplier.strip() and inn_sup:
        supplier = _entity_name_near_inn(full, inn_sup) or supplier

    out: Dict[str, Any] = {
        "Банк получателя": bank,
        "Получатель": recipient,
        "ИНН поставщика": inn_sup,
        "КПП поставщика": kpp_sup,
        "ИНН покупателя": inn_buy,
        "КПП покупателя": kpp_buy,
        "БИК": bik_str,
        "Счет": acc_str,
        "Поставщик": supplier,
        "Покупатель": buyer,
        "Номер счета": inv_num,
        "Дата счета": inv_date,
        "Итого": itogo,
        "Товары": names_join,
        "Количество": qty_join,
        "Ед.Из": unit_join,
        "Цена": price_join,
        "Сумма": sum_join,
        "items": items,
    }
    return out


def enrich_fields_from_regex_fallback(text: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Если merge/нейросеть оставили пустыми ключевые поля — дополняет из эвристик по полному тексту.
    Нужно для neural_trained и для случаев, когда seq2seq «теряет» дату или сумму.
    """
    if not (text or "").strip():
        return fields
    rx = extract_invoice_fields(text)
    out = dict(fields)
    for k in FIELD_LABELS:
        if str(out.get(k) or "").strip():
            continue
        v = rx.get(k)
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            out[k] = v
        elif not isinstance(v, str) and str(v).strip():
            out[k] = v
    if not out.get("items") and rx.get("items"):
        out["items"] = list(rx["items"])
    return out


def fields_to_flat_rows(filename: str, fields: Dict[str, Any]) -> List[Dict[str, str]]:
    """Для отображения в таблице: одна строка на поле."""
    rows = []
    if fields and "ИНН Поставщика" in fields:
        key_order = list(PUBLIC_FIELD_KEYS)
    else:
        key_order = list(FIELD_LABELS)
    for key in key_order:
        if key not in fields:
            continue
        val = fields.get(key, "")
        if isinstance(val, str):
            rows.append({"Файл": filename, "Поле": key, "Значение": val})
    items = fields.get("items") or []
    if items:
        rows.append(
            {
                "Файл": filename,
                "Поле": "Позиции (таблица)",
                "Значение": json.dumps(items, ensure_ascii=False),
            },
        )
    return rows
