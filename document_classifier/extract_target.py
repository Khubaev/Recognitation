"""Сериализация целевого JSON для обучения извлечения и разбор ответа модели."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .invoice_fields import (
    FIELD_LABELS,
    _is_bad_bank_name,
    _is_bad_recipient,
    _is_likely_bank_rs_number,
    is_itogo_amount_in_words,
    joined_fields_from_items,
)


def _from_nested_invoice_schema(d: Dict[str, Any]) -> Dict[str, Any]:
    """JSON вида { recipient, buyer, items, basis } → плоские ключи FIELD_LABELS."""
    rec = d.get("recipient")
    buyer = d.get("buyer")
    if not isinstance(rec, dict):
        rec = {}
    if not isinstance(buyer, dict):
        buyer = {}
    inv_no = str(
        rec.get("number")
        or rec.get("invoiceNumber")
        or d.get("number")
        or d.get("invoiceNumber")
        or d.get("Номер счета")
        or "",
    ).strip()
    inv_dt = str(
        rec.get("date")
        or rec.get("invoiceDate")
        or d.get("date")
        or d.get("Дата счета")
        or "",
    ).strip()
    tot = str(
        rec.get("total")
        or rec.get("sum")
        or d.get("total")
        or d.get("Итого")
        or "",
    ).strip()

    out: Dict[str, Any] = {
        "Банк получателя": str(rec.get("bankName") or "").strip(),
        "Получатель": str(rec.get("name") or "").strip(),
        "ИНН поставщика": str(rec.get("inn") or "").strip(),
        "КПП поставщика": str(rec.get("kpp") or "").strip(),
        "ИНН покупателя": str(buyer.get("inn") or "").strip(),
        "КПП покупателя": str(buyer.get("kpp") or "").strip(),
        "БИК": str(rec.get("bik") or "").strip(),
        "Счет": str(rec.get("account") or "").strip(),
        "Поставщик": str(rec.get("name") or "").strip(),
        "Покупатель": str(buyer.get("name") or "").strip(),
        "Номер счета": inv_no,
        "Дата счета": inv_dt,
        "Итого": tot,
    }
    raw_items = d.get("items") or []
    clean: List[Dict[str, str]] = []
    if isinstance(raw_items, list):
        for it in raw_items:
            if isinstance(it, dict):
                clean.append(
                    {
                        "name": str(it.get("name", "")),
                        "qty": str(it.get("quantity", it.get("qty", ""))),
                        "unit": str(it.get("unit", "")),
                        "price": str(it.get("price", "")),
                        "sum": str(it.get("sum", "")),
                    },
                )
    out["items"] = clean
    if clean:
        out.update(joined_fields_from_items(clean))
    else:
        for k in ("Товары", "Количество", "Ед.Из", "Цена", "Сумма"):
            out.setdefault(k, "")
    return out


def normalize_sidecar_payload(raw: Any) -> Dict[str, Any]:
    """Корень JSON — объект или [ { … } ]; вложенная схема recipient/buyer → плоский словарь."""
    if raw is None:
        return {}
    if isinstance(raw, list):
        if not raw:
            return {}
        raw = raw[0]
    if not isinstance(raw, dict):
        return {}
    if "recipient" in raw or "buyer" in raw:
        return _from_nested_invoice_schema(raw)
    return raw


def canonicalize_extract_labels(d: Any) -> Dict[str, Any]:
    """
    Приводит ключи разметки (в т.ч. «ИНН Покупателя» / «КПП Продавца») к FIELD_LABELS.
    Если в JSON два поля ИНН покупателя в разных формулировках — первое трактуем как ИНН поставщика.
    """
    d = normalize_sidecar_payload(d)
    if not d:
        return {}
    raw: Dict[str, Any] = {}
    for k, v in d.items():
        if str(k).startswith("_"):
            continue
        raw[str(k)] = v

    items = raw.pop("items", None)

    if "ИНН Покупателя" in raw and "ИНН Покупатель" in raw:
        raw["ИНН поставщика"] = str(raw["ИНН Покупателя"]).strip()
        raw["ИНН покупателя"] = str(raw["ИНН Покупатель"]).strip()
    elif "ИНН Покупатель" in raw:
        raw.setdefault("ИНН покупателя", str(raw["ИНН Покупатель"]).strip())
    elif "ИНН Покупателя" in raw:
        raw.setdefault("ИНН покупателя", str(raw["ИНН Покупателя"]).strip())

    for old, new in (
        ("КПП Продавца", "КПП поставщика"),
        ("КПП Поставщика", "КПП поставщика"),
        ("КПП Покупатель", "КПП покупателя"),
        ("КПП Покупателя", "КПП покупателя"),
        ("ИНН Поставщика", "ИНН поставщика"),
        ("ИНН Продавца", "ИНН поставщика"),
        ("ИНН продавца", "ИНН поставщика"),
        ("Наименование покупателя", "Покупатель"),
        ("Номер счёта", "Номер счета"),
        ("Номер документа", "Номер счета"),
        ("Дата документа", "Дата счета"),
        ("Итого к оплате", "Итого"),
        ("Всего к оплате", "Итого"),
        ("Сумма к оплате", "Итого"),
        ("СуммаИтого", "Итого"),
        ("НомерСчета", "Номер счета"),
        ("ДатаСчета", "Дата счета"),
    ):
        if old in raw:
            raw.setdefault(new, str(raw[old]).strip())

    if "Наименование поставщика" in raw:
        raw.setdefault("Поставщик", str(raw["Наименование поставщика"]).strip())

    if "ИНН" in raw:
        parts = [p.strip() for p in str(raw["ИНН"]).split(";") if p.strip()]
        if len(parts) >= 1:
            raw.setdefault("ИНН поставщика", parts[0])
        if len(parts) >= 2:
            raw.setdefault("ИНН покупателя", parts[1])

    if "КПП" in raw:
        parts = [p.strip() for p in str(raw["КПП"]).split(";") if p.strip()]
        if len(parts) >= 1:
            raw.setdefault("КПП поставщика", parts[0])
        if len(parts) >= 2:
            raw.setdefault("КПП покупателя", parts[1])

    out: Dict[str, Any] = {k: "" for k in FIELD_LABELS}
    for k in FIELD_LABELS:
        v = raw.get(k)
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            out[k] = json.dumps(v, ensure_ascii=False) if k != "items" else ""
        else:
            out[k] = str(v).strip()

    if isinstance(items, list):
        out["items"] = items
    else:
        out["items"] = []
    return out


def fields_to_target_json(fields: Dict[str, Any]) -> str:
    """Компактная строка JSON для seq2seq-цели."""
    merged = canonicalize_extract_labels(fields)
    payload: Dict[str, Any] = {k: merged.get(k, "") for k in FIELD_LABELS}
    payload["items"] = merged.get("items") or []
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_model_json(raw: str) -> Dict[str, Any]:
    """Достаёт JSON из ответа модели (могут быть лишние символы)."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    start = raw.find("{")
    if start == -1:
        return {}
    depth = 0
    end = -1
    for i, c in enumerate(raw[start:], start=start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return {}
    chunk = raw[start:end]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        chunk = re.sub(r",\s*}", "}", chunk)
        chunk = re.sub(r",\s*]", "]", chunk)
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return {}


def _looks_like_contract_or_payment_ref(s: str) -> bool:
    """Ссылка на договор/платёж вместо номенклатуры из таблицы."""
    if not s or not str(s).strip():
        return False
    s = str(s).strip()
    if len(s) < 8:
        return False
    if re.search(r"ЧлФ|/20\d{2}/|договор\s*№|№\s*\d+\s*от\s+\d", s, re.I):
        return True
    # «07/23 от 10.03.2023», договор с датой
    if re.search(r"\d{1,3}/\d{2,4}\s+от\s+\d{2}\.\d{2}\.\d{2,4}", s):
        return True
    if re.search(r"№?\s*\d+\s*от\s+\d{2}\.\d{2}\.\d{4}", s, re.I):
        return True
    return False


def _prefer_longer_party(ns: str, rs: str) -> str:
    """Если одна строка — подстрока другой (например «СТЕЙТ» в «ООО СТЕЙТ»), берём длиннее."""
    ns, rs = (ns or "").strip(), (rs or "").strip()
    if not rs:
        return ns
    if not ns:
        return rs
    if ns in rs and len(rs) > len(ns) + 1:
        return rs
    if rs in ns and len(ns) > len(rs) + 1:
        return ns
    return ns


def _merge_party_recipient(ns: str, rs: str, supplier: str) -> str:
    """Не оставлять нейросетевой мусор («оплаты»), если есть regex или поставщик."""
    ns, rs, supplier = (ns or "").strip(), (rs or "").strip(), (supplier or "").strip()
    good_ns = bool(ns) and not _is_bad_recipient(ns)
    good_rs = bool(rs) and not _is_bad_recipient(rs)
    good_sup = bool(supplier) and not _is_bad_recipient(supplier)
    if good_ns and good_rs:
        return _prefer_longer_party(ns, rs)
    if good_ns:
        return ns or rs
    if good_rs:
        return rs or ns
    if good_sup:
        return supplier
    return ns or rs


def _joined_item_names_blob(items: List[Any]) -> str:
    if not items:
        return ""
    names: List[str] = []
    for it in items:
        if isinstance(it, dict):
            names.append(str(it.get("name", "") or "").strip())
    return " | ".join(names).strip()


def merge_extracted(neural: Dict[str, Any], regex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Смешивание нейросети и эвристик: для контрагентов/банка — подстраховка и полные названия;
    табличная часть — приоритет у эвристик, если у сети «договор» вместо номенклатуры или строк меньше.
    """
    goods_keys = ("Товары", "Количество", "Ед.Из", "Цена", "Сумма")
    party_keys = ("Поставщик", "Покупатель", "Получатель")

    out: Dict[str, Any] = {}

    for k in FIELD_LABELS:
        if k in goods_keys:
            continue
        nv = neural.get(k)
        rv = regex.get(k, "")
        ns = nv.strip() if isinstance(nv, str) else ("" if nv is None else str(nv))
        rs = rv.strip() if isinstance(rv, str) else str(rv or "")

        if k == "Банк получателя":
            if _is_bad_bank_name(ns) and rs:
                out[k] = rs
            elif ns and not _is_bad_bank_name(ns):
                out[k] = _prefer_longer_party(ns, rs)
            else:
                out[k] = rs or ns
            continue

        if k in party_keys:
            if k == "Получатель":
                sup_for_r = (regex.get("Поставщик") or "").strip()
                out[k] = _merge_party_recipient(ns, rs, sup_for_r)
            elif _is_bad_recipient(ns) and rs:
                # Иначе _prefer_longer_party оставляет мусор LLM («Z,N - заказной товар…»)
                out[k] = rs
            elif ns and rs:
                out[k] = _prefer_longer_party(ns, rs)
            else:
                out[k] = ns or rs
            continue

        if k == "Итого":
            if is_itogo_amount_in_words(ns) and rs:
                out[k] = rs
                continue

        if k == "Номер счета":
            if _is_likely_bank_rs_number(ns) and rs and not _is_likely_bank_rs_number(rs):
                out[k] = rs
                continue

        if isinstance(nv, str) and nv.strip():
            out[k] = nv.strip()
        elif nv is not None and not isinstance(nv, str):
            out[k] = str(nv)
        else:
            out[k] = rs

    n_items = neural.get("items") if isinstance(neural.get("items"), list) else []
    r_items = regex.get("items") if isinstance(regex.get("items"), list) else []
    nt = (neural.get("Товары") or "").strip()
    nn_blob = _joined_item_names_blob(n_items) or nt

    choose_regex_items = False
    if len(r_items) > len(n_items):
        choose_regex_items = True
    elif _looks_like_contract_or_payment_ref(nn_blob) and len(r_items) > 0:
        choose_regex_items = True
    elif _looks_like_contract_or_payment_ref(nt) and len(r_items) > 0:
        choose_regex_items = True
    elif len(n_items) == 0 and len(r_items) > 0:
        choose_regex_items = True

    if choose_regex_items:
        out["items"] = r_items
    elif len(n_items) > 0:
        out["items"] = n_items
    else:
        out["items"] = r_items

    if out.get("items"):
        out.update(joined_fields_from_items(out["items"]))
    else:
        for g in goods_keys:
            nv = neural.get(g)
            rv = regex.get(g, "")
            ns = nv.strip() if isinstance(nv, str) else ""
            rs = rv.strip() if isinstance(rv, str) else str(rv or "")
            if _looks_like_contract_or_payment_ref(ns) and rs:
                out[g] = rs
            elif isinstance(nv, str) and nv.strip():
                out[g] = nv.strip()
            else:
                out[g] = rs

    # Нейросеть заполнила позиции ссылкой на договор; regex-таблицы нет — берём плоские поля regex
    if out.get("items"):
        jt = _joined_item_names_blob(out["items"]) or (out.get("Товары") or "")
        if _looks_like_contract_or_payment_ref(jt) and not choose_regex_items:
            if r_items:
                out["items"] = r_items
                out.update(joined_fields_from_items(r_items))
            else:
                out["items"] = []
                for g in goods_keys:
                    rv = regex.get(g, "")
                    rs = rv.strip() if isinstance(rv, str) else str(rv or "")
                    nv = neural.get(g)
                    ns = nv.strip() if isinstance(nv, str) else str(nv or "")
                    if rs and not _looks_like_contract_or_payment_ref(rs):
                        out[g] = rs
                    elif rs:
                        out[g] = rs
                    elif ns and not _looks_like_contract_or_payment_ref(ns):
                        out[g] = ns
                    else:
                        out[g] = ""

    return out


def normalize_parsed(d: Dict[str, Any]) -> Dict[str, Any]:
    """Алиасы ключей из ответа модели + нормализация items."""
    if not d:
        return {k: "" for k in FIELD_LABELS} | {"items": []}
    base = canonicalize_extract_labels(d)
    items = base.get("items") or []
    clean: List[Dict[str, str]] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                clean.append(
                    {
                        "name": str(it.get("name", "")),
                        "qty": str(it.get("qty", "")),
                        "unit": str(it.get("unit", "")),
                        "price": str(it.get("price", "")),
                        "sum": str(it.get("sum", "")),
                    },
                )
    base["items"] = clean
    return base
