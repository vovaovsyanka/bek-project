"""
Основная логика обработки запроса через цепочку классификаторов.

Схема:
1. Общий BERT → безопасен? → LLM → ответ
              ↓ нет
2. Категориальный BERT (3/4/5) → целевая категория?
              ↓ нет → отказ ("Категория запроса не соответствует выбранной")
              ↓ да
3. Общий классификатор по уровню (TF-IDF=1 / LSTM=2 / BERT=3)
              ↓ опасен → "Модель поняла что запрос опасный и его нужно усилить"
              ↓ безопасен → LLM → ответ
"""

import random
from typing import Optional

from llm_client import LLMClient
from base_classifier import RuBert, LSTM, TF_IDF
from category_classifier import CategoryClassifier


class ProcessingPipeline:
    """Пайплайн обработки запроса через каскад классификаторов."""

    def __init__(self):
        # Клиент LLM
        self.llm_client = LLMClient()

        # 1. Общий классификатор BERT (опасный / безопасный)
        self.general_bert = RuBert()

        # 2. Категориальные классификаторы (ленивая инициализация при первом использовании)
        self._category_classifiers: dict[int, CategoryClassifier] = {}

        # 3. Общие классификаторы по уровню сложности
        self.level_classifiers = {
            1: TF_IDF(),    # Уровень 1: TF-IDF + линейная модель
            2: LSTM(),      # Уровень 2: LSTM
            3: RuBert(),    # Уровень 3: BERT
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
            difficulty: Уровень сложности (1-TF-IDF, 2-LSTM, 3-BERT)

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
            return f"Ошибка: неизвестный уровень сложности {difficulty}. Доступные: 1, 2, 3."

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
        # Используем ghost_id=1 (первый призрак) и пароль из конфига
        # Вы можете изменить логику выбора призрака при необходимости
        ghost_id = 1
        current_password = "АПЛОДИСМЕНТЫ"  # Пароль первого призрака из config.py

        try:
            response = self.llm_client.get_response(query, ghost_id, current_password)
            return response
        except Exception as e:
            return f"Ошибка при обращении к LLM: {str(e)}"
