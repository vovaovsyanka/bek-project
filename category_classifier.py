"""
BERT классификатор для определения принадлежности запроса к целевой категории.
Для каждой категории (3, 4, 5) загружается отдельная дообученная модель.
"""

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CategoryClassifier:
    """
    Классификатор на основе BERT, определяющий,
    принадлежит ли запрос к целевой категории (target) или нет (other).
    """

    # === НАСТРОЙКИ ПУТЕЙ К МОДЕЛЯМ ===
    CATEGORY_MODEL_PATHS = {
        3: "models/ru-en-RoSBERTa_3",
        4: "models/ru-en-RoSBERTa_4",
        5: "models/ru-en-RoSBERTa_5",
    }

    def __init__(self, category: int):
        """
        Args:
            category: Номер целевой категории (3, 4 или 5)
        """
        if category not in [3, 4, 5]:
            raise ValueError(f"Категория должна быть 3, 4 или 5. Получена: {category}")

        self.category = category
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = self.CATEGORY_MODEL_PATHS.get(category)

        if model_path and os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/ru-en-RoSBERTa")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            print(f"[CategoryClassifier] Внимание: модель для категории {category} не найдена по пути {model_path}")
            print(f"[CategoryClassifier] Используется заглушка — случайная классификация.")
            self.tokenizer = None
            self.model = None

        if self.model:
            self.model.to(self.device)
            self.model.eval()

    def _preprocess_text(self, text: str) -> str:
        """Базовая очистка: нижний регистр, удаление знаков препинания, схлопывание пробелов."""
        _PUNCT_CHARS = (
            ".,!?;:()[]{}-_%$#@+=/\*&~"
            "\u00ab\u00bb"        # « »
            "\u201e\u201c\u201d"  # „ “ ”
            "\u2018\u2019"        # ‘ ’
            "\u2026"              # …
            "\u0022\u0027"        # " '
        )
        _punct_pattern = re.compile("[" + re.escape(_PUNCT_CHARS) + "]")
        _multi_space = re.compile(r"\s+")

        text = str(text).strip().lower()
        text = _punct_pattern.sub(" ", text)
        text = _multi_space.sub(" ", text).strip()
        return text

    def classify(self, text: str) -> bool:
        """
        Классифицирует запрос.

        Returns:
            True — запрос ПРИНАДЛЕЖИТ целевой категории
            False — запрос НЕ принадлежит целевой категории
        """
        if self.model is None or self.tokenizer is None:
            return True

        cleaned = self._preprocess_text(text)

        inputs = self.tokenizer(
            cleaned,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = seld.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        return torch.argmax(logits, dim=-1).item()

    def get_name(self) -> str:
        return f"CategoryBERT_{self.category}"
