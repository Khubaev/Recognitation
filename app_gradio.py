"""
Веб-интерфейс: загрузка нескольких PDF/Excel, поочерёдная классификация, таблица результатов.
Запуск: py -3 app_gradio.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import pandas as pd

from document_classifier.inference import DocumentClassifier
from document_classifier.invoice_fields import fields_to_flat_rows

DEFAULT_CKPT = Path("checkpoints/doc_classifier")


def _to_path(f) -> Path:
    if isinstance(f, str):
        return Path(f)
    name = getattr(f, "name", None)
    if name:
        return Path(name)
    return Path(str(f))


def classify_batch(
    files,
    checkpoint_str: str,
    fields_mode_ui: str,
    no_neural_extract_ckpt: bool,
    text_extract_mode_ui: str,
):
    checkpoint = Path(checkpoint_str)
    empty = pd.DataFrame(
        columns=["№", "Файл", "Тип документа", "Уверенность", "Символов извлечено", "Замечание"],
    )
    empty_fields = pd.DataFrame(columns=["Файл", "Поле", "Значение"])

    if not files:
        yield empty, "Загрузите один или несколько файлов (PDF, xlsx, xls).", empty_fields
        return

    paths = [_to_path(f) for f in files]

    try:
        if no_neural_extract_ckpt:
            clf = DocumentClassifier(
                checkpoint,
                use_neural_extract=False,
                fields_mode="regex_only",
            )
        elif fields_mode_ui == "neural_only":
            clf = DocumentClassifier(checkpoint, fields_mode="neural_only")
        elif fields_mode_ui == "merge":
            clf = DocumentClassifier(checkpoint, fields_mode="merge")
        else:
            clf = DocumentClassifier(checkpoint, fields_mode="regex_only")
    except Exception as e:
        yield empty, (
            f"**Не удалось загрузить модель** из `{checkpoint}`.\n\n"
            f"Ошибка: `{e}`\n\n"
            "Сначала обучите модель, например:\n"
            "`py -3 train.py --labeled-root data\\labeled_files --valid-ratio 0.15`"
        ), empty_fields
        return

    rows: list[dict] = []
    field_rows: list[dict] = []
    n = len(paths)
    for i, p in enumerate(paths, start=1):
        r = clf.predict_file(p, text_extract_mode=text_extract_mode_ui)
        err = (r.get("error") or "").strip()
        note = err if err else "—"
        label = r.get("label") or "—"
        conf = r.get("confidence")
        conf_s = f"{conf:.4f}" if isinstance(conf, float) else "—"
        chars = r.get("extracted_chars", 0)
        rows.append(
            {
                "№": i,
                "Файл": p.name,
                "Тип документа": label,
                "Уверенность": conf_s,
                "Символов извлечено": chars,
                "Замечание": note,
            },
        )
        fields = r.get("fields") or {}
        field_rows.extend(fields_to_flat_rows(p.name, fields))
        df = pd.DataFrame(rows)
        df_f = pd.DataFrame(field_rows)
        yield df, f"Обработано **{i}** из **{n}**: `{p.name}` → **{label}** (реквизиты обновлены)", df_f

    yield (
        pd.DataFrame(rows),
        f"**Готово.** Обработано файлов: **{n}**.",
        pd.DataFrame(field_rows),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Временная публичная ссылка Gradio")
    args = ap.parse_args()
    ckpt = str(args.checkpoint.resolve())

    with gr.Blocks(title="Классификация первичных документов") as demo:
        gr.Markdown(
            "### Классификация и извлечение реквизитов\n"
            "Загрузите **PDF** (с текстовым слоем) или **Excel** — файлы обрабатываются **по очереди**. "
            "По умолчанию: **нейросеть** `checkpoints/invoice_extract` **и эвристики** — regex подставляется, "
            "если поле модель не заполнила. Режим «только нейросеть» отключает regex (кроме случая полностью "
            "пустого ответа — тогда всё равно показываются эвристики).",
        )
        ckpt_box = gr.Textbox(value=ckpt, label="Каталог чекпоинта", visible=False)
        fields_mode = gr.Radio(
            choices=[
                ("Только дообученная модель извлечения", "neural_only"),
                ("Нейросеть + эвристики (подстраховка)", "merge"),
                ("Только эвристики (regex)", "regex_only"),
            ],
            value="merge",
            label="Как заполнять реквизиты",
        )
        no_neural_cb = gr.Checkbox(
            label="Не загружать invoice_extract (только эвристики)",
            value=False,
        )
        text_extract_mode = gr.Radio(
            choices=[
                ("Авто (PDF/картинки — Yandex Vision при OCR_YANDEX_*; Word/Excel/текст — локально)", "auto"),
                ("Только локально (PyMuPDF/Office/текст; без OCR; картинки — ошибка)", "local"),
                ("Всегда OCR Yandex для PDF и изображений", "ocr"),
            ],
            value="auto",
            label="Извлечение текста до моделей",
        )
        files_in = gr.File(
            label="Документы",
            file_count="multiple",
            file_types=[".pdf", ".docx", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"],
        )
        run_btn = gr.Button("Классифицировать", variant="primary")
        status = gr.Markdown(value="")
        table = gr.Dataframe(label="Классификация", wrap=True, interactive=False)
        gr.Markdown("#### Извлечённые параметры")
        fields_table = gr.Dataframe(
            label="Реквизиты и позиции",
            wrap=True,
            interactive=False,
        )

        run_btn.click(
            fn=classify_batch,
            inputs=[files_in, ckpt_box, fields_mode, no_neural_cb, text_extract_mode],
            outputs=[table, status, fields_table],
        )

    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
