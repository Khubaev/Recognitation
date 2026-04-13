# Типы первичных документов (порядок фиксирован — совпадает с id класса)
DOC_LABELS = [
    "СчетНаОплату",
    "АктВыполненныхРабот",
    "ТоварнаяНакладная",
    "УПД",
    "СчетФактура",
    "Договор",
    "Прочее",
]

LABEL2ID = {name: i for i, name in enumerate(DOC_LABELS)}
ID2LABEL = {i: name for i, name in enumerate(DOC_LABELS)}

DEFAULT_MODEL_NAME = "cointegrated/rubert-tiny2"
