"""
Основная логика обработки запроса через цепочку классификаторов.

Схема:
1. Общий BERT → безопасен? → LLM → ответ
              ↓ нет
2. Категориальный BERT (3/4/5) → целевая категория?
              ↓ нет → отказ ("Категория запроса не соответствует выбранной")
              ↓ да
3. Общий классификатор по уровню:
   1 - Dictionary (словарь)
   2 - TF-IDF
   3 - LSTM
   4 - BERT
              ↓ опасен → "Модель поняла что запрос опасный и его нужно усилить"
              ↓ безопасен → LLM → ответ
"""

import random
from typing import Optional

from llm_client import LLMClient
from base_classifier import MultilingualBert, LSTM, TF_IDF, Dictionary, RoSBERTa
from category_classifier import CategoryClassifier


class ProcessingPipeline:
    """Пайплайн обработки запроса через каскад классификаторов."""

    def __init__(self):
        # Клиент LLM
        self.llm_client = LLMClient()

        # 1. Общий классификатор BERT (опасный / безопасный)
        self.general_bert = RoSBERTa()

        # 2. Категориальные классификаторы (ленивая инициализация при первом использовании)
        self._category_classifiers: dict[int, CategoryClassifier] = {}

        # 3. Общие классификаторы по уровню сложности
        self.level_classifiers = {
            1: Dictionary(),   # уровень 1 - словарь
            2: TF_IDF(),       # уровень 2 - TF-IDF
            3: LSTM(),         # уровень 3 - LSTM
            4: MultilingualBert(),       # уровень 4 - BERT
        }

    def _get_category_classifier(self, category: int) -> CategoryClassifier:
        """Получает (или создаёт) категориальный классификатор."""
        if category not in self._category_classifiers:
            self._category_classifiers[category] = CategoryClassifier(category)
        return self._category_classifiers[category]

    def process(self, query: str, category: int, difficulty: int) -> str:
        """
        Основной метод обработки запроса.

        Args:
            query: Текст запроса пользователя
            category: Целевая категория (3, 4 или 5)
            difficulty: Уровень сложности (1-Dictionary, 2-TF-IDF, 3-LSTM, 4-BERT)

        Returns:
            Текстовый ответ для пользователя
        """
        # ============================================================
        # ШАГ 1: Общий BERT классификатор (опасный / безопасный)
        # ============================================================
        general_result = self.general_bert.classify(query)

        if general_result == 0:
            # Запрос БЕЗОПАСНЫЙ → сразу к LLM
            return self._call_llm(query)

        # ============================================================
        # ШАГ 2: Запрос ОПАСНЫЙ → проверяем категорию
        # ============================================================
        cat_classifier = self._get_category_classifier(category)
        is_target_category = cat_classifier.classify(query)

        if not is_target_category:
            # Запрос не из целевой категории
            return (
                f"Категория вашего запроса не соответствует выбранной "
                f"(вы выбрали категорию {category}). "
                f"Попробуйте выбрать другую категорию или переформулировать запрос."
            )

        # ============================================================
        # ШАГ 3: Запрос из целевой категории → общий классификатор по уровню
        # ============================================================
        if difficulty not in self.level_classifiers:
            return f"Ошибка: неизвестный уровень сложности {difficulty}. Доступные: 1, 2, 3, 4."

        level_classifier = self.level_classifiers[difficulty]
        level_result = level_classifier.classify(query)

        if level_result == 1:
            # Запрос ОПАСНЫЙ по мнению классификатора уровня
            return (
                f"Модель ({level_classifier.get_name()}) поняла, что запрос опасный "
                f"и его нужно усилить. Попробуйте обойти защиту этого уровня."
            )

        # ============================================================
        # ШАГ 4: Запрос БЕЗОПАСНЫЙ по мнению классификатора уровня → LLM
        # ============================================================
        return self._call_llm(query)

    def _call_llm(self, query: str) -> str:
        """Вызывает LLM для генерации ответа."""
        ghost_id = 1
        try:
            response = self.llm_client.get_response(query, ghost_id, current_password=None)
            return response
        except Exception as e:
            return f"Ошибка при обращении к LLM: {str(e)}"