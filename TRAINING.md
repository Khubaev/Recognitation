# Обучение и проверка моделей

В проекте **две независимые модели**:

| Задача | Скрипт | Выход (чекпоинт) |
|--------|--------|------------------|
| Классификация типа документа | `train.py` | `checkpoints/doc_classifier` |
| Извлечение реквизитов (seq2seq) | `train_extract.py` | `checkpoints/invoice_extract` |

Инференс по умолчанию подгружает **оба** каталога: классификатор из `--checkpoint` (или значения по умолчанию), извлечение — из `checkpoints/invoice_extract`, если там есть `config.json`.

---

## Подготовка окружения

Из корня репозитория (Windows):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Дальше команды с `py -3` можно запускать с активированным venv или через `py -3` напрямую.

---

## 1. Обучение классификатора документов

**Вход:** пары «текст → метка класса».

- Файл `data/train.jsonl`: строки вида `{"text": "...", "label": 0}` (индекс класса из `document_classifier.config.DOC_LABELS`).
- Или каталог **`--labeled-root`**: подпапки с **именами классов**, внутри PDF / xlsx / xls — текст извлекается автоматически.

**Пример запуска** (разметка в `data/labeled_files`, часть выборки — валидация):

```powershell
py -3 train.py --labeled-root data\labeled_files --valid-ratio 0.15 --out checkpoints\doc_classifier
```

**Полезные параметры:**

- `--train data\train.jsonl` — дополнительно подмешать jsonl к обучению.
- `--valid data\valid.json` — явная валидация (json/jsonl), если не используете только `--valid-ratio`.
- `--epochs`, `--batch`, `--lr`, `--model` — см. `py -3 train.py --help`.

**Результат:** в конце обучения модель и токенайзер сохраняются в `--out` (по умолчанию `checkpoints/doc_classifier`). При наличии валидации в логе печатается **classification_report** (precision/recall/F1 по классам).

---

## 2. Данные и обучение модели извлечения полей

### 2.1. Сбор датасета

Скрипт **`prepare_extract_dataset.py`** собирает `data/extract_train.jsonl` (строки `{"text": "...", "target": "<компактный JSON>"}`):

- подмешивает `data/extract_seed.jsonl` (ручные примеры);
- обходит PDF/xlsx/xls в папке счетов (по умолчанию `data/labeled_files/СчетНаОплату`);
- если рядом с документом лежит одноимённый `.json` с разметкой — он используется как цель; иначе цель строится эвристикой `extract_invoice_fields`.

```powershell
py -3 prepare_extract_dataset.py
```

Параметры: `--invoices-dir`, `--seed`, `--out` — см. `py -3 prepare_extract_dataset.py --help`.

### 2.2. Запуск обучения извлечения

```powershell
py -3 train_extract.py --train data\extract_train.jsonl --out checkpoints\invoice_extract
```

**Полезные параметры:** `--epochs` (по умолчанию 30), `--batch`, `--lr`, `--model` (по умолчанию `google/flan-t5-small`), `--valid-ratio` (по умолчанию 0.15; при очень малой выборке валидация может отключиться). См. `py -3 train_extract.py --help`.

**Результат:** чекпоинт в `--out` (по умолчанию `checkpoints/invoice_extract`). При валидации Trainer сохраняет лучшую по `eval_loss` эпоху (`load_best_model_at_end`).

---

## 3. Как проверять качество

### 3.1. Командная строка: `predict.py`

Нужен обученный **классификатор** (`checkpoints/doc_classifier`). Модель извлечения подхватится из `checkpoints/invoice_extract`, если каталог существует и в нём есть `config.json`.

```powershell
py -3 predict.py --document path\to\file.pdf --fields
```

Или текст:

```powershell
py -3 predict.py --text "фрагмент текста счёта..." --fields
```

**Режимы полей** (как в коде `DocumentClassifier`):

- по умолчанию — **нейросеть + эвристики** (merge);
- `--merge-fields` — явно то же;
- `--regex-only-fields` — только regex;
- `--no-neural-fields` — не загружать seq2seq, только эвристики.

Смотрите в выводе: строка **«Тип: …»** (классификация) и блок **«--- Реквизиты ---»** при `--fields`.

### 3.2. Веб-интерфейс Gradio

```powershell
py -3 app_gradio.py
```

Откройте в браузере адрес (по умолчанию `http://127.0.0.1:7860`). Загрузите PDF/Excel: таблица покажет тип документа, уверенность и плоский список полей. В настройках можно выбрать **только нейросеть**, **нейросеть + эвристики** или **только regex**.

### 3.3. На что ориентироваться

- **Классификатор:** отчёт при валидации в конце `train.py`; на новых файлах — стабильность метки и `confidence` в `predict.py` / Gradio.
- **Извлечение:** полнота и точность полей на реальных счетах; при малых данных seq2seq часто ошибается — режим **merge** и эвристики в коде как раз подстраховывают. Улучшение обычно даёт **больше размеченных пар** в `extract_train.jsonl` (качественные sidecar `.json` рядом с PDF).

---

## 4. Типичный порядок работ

1. Разметить документы в `data/labeled_files\...` (и при необходимости пополнить `data/extract_seed.jsonl`).
2. `py -3 train.py --labeled-root data\labeled_files --valid-ratio 0.15`
3. `py -3 prepare_extract_dataset.py`
4. `py -3 train_extract.py`
5. Проверка: `py -3 predict.py --document ... --fields` и/или `py -3 app_gradio.py`

Если чекпоинт извлечения не нужен, достаточно классификатора: `predict.py` и Gradio будут работать с **regex** для полей (`--no-neural-fields` или отсутствие `checkpoints/invoice_extract`).
