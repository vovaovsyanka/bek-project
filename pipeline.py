"""
Основная логика обработки запроса через цепочку классификаторов.

Схема (объединённый process):
  0. check_password(category)  — инициализируем/возвращаем пароль для категории
  0b. Проверяем совпадение text с паролем → passed
      passed=True  → rotate password, вернуть {"text": "", "passed": True}
  1. Общий BERT → безопасен? → LLM(current_password) → ответ
     ↓ нет
  2. Категориальный BERT → целевая категория?
     ↓ нет → отказ
     ↓ да
  3. TF-IDF (difficulty фиксирован = GAME_DIFFICULTY=2)
     ↓ опасен → "нужно усилить"
     ↓ безопасен → LLM(current_password) → ответ
"""

import random
from typing import Optional

from llm_client import LLMClient
from base_classifier import MultilingualBert, LSTM, TF_IDF, Dictionary, RoSBERTa
from category_classifier import CategoryClassifier
from config import GHOSTS, LEVEL_TO_CATEGORY, GAME_DIFFICULTY

# Глобальный словарь: хранит текущий пароль для каждой категории {category: password}
_current_passwords: dict[int, str] = {}


def check_password(category: int, force_new: bool = False) -> str:
    """
    Возвращает текущий пароль для данной категории.

    Логика:
      - Если force_new=True  → всегда выбираем новый случайный пароль.
      - Если слово уже есть  → возвращаем существующее (пропускаем выбор).
      - Если слова нет       → выбираем случайный из GHOSTS[1]["passwords"].

    Args:
        category:  Категория (3, 4 или 5)
        force_new: Принудительно сменить пароль (вызывается после passed=True)

    Returns:
        Текущий (или новый) пароль для категории
    """
    passwords = GHOSTS[category]["passwords"]

    if force_new or category not in _current_passwords:
        _current_passwords[category] = random.choice(passwords)

    return _current_passwords[category]


class ProcessingPipeline:
    """Пайплайн обработки запроса через каскад классификаторов."""

    def __init__(self):
        self.llm_client = LLMClient()

        # Общий BERT классификатор (опасный / безопасный)
        self.general_bert = RoSBERTa()

        # Категориальные классификаторы (ленивая инициализация)
        self._category_classifiers: dict[int, CategoryClassifier] = {}

        # Классификаторы по уровню сложности
        self.level_classifiers = {
            1: Dictionary(),
            2: TF_IDF(),
            3: LSTM(),
            4: MultilingualBert(),
        }

    def _get_category_classifier(self, category: int) -> CategoryClassifier:
        if category not in self._category_classifiers:
            self._category_classifiers[category] = CategoryClassifier(category)
        return self._category_classifiers[category]

    def process(self, text: str, level: str) -> dict:
        """
        Единственный публичный метод пайплайна.

        Args:
            text:  Текст пользователя
            level: Игровой уровень (tony_stark / subnautica / cyberpunk)

        Returns:
            {"text": str, "passed": bool}
        """
        # ── Определяем категорию по уровню ──────────────────────────────────
        category = LEVEL_TO_CATEGORY.get(level)
        if category is None:
            return {
                "text": (
                    f"Неизвестный уровень '{level}'. "
                    f"Допустимые значения: {', '.join(LEVEL_TO_CATEGORY.keys())}."
                ),
                "passed": False,
            }

        try:
            # ── ШАГ 0: Инициализируем/получаем пароль для категории ─────────
            current_password = check_password(category)

            # ── ШАГ 0b: Проверяем совпадение с паролем ──────────────────────
            passed: bool = text.strip().upper() == current_password.upper()

            if passed:
                # Пароль угадан — ротируем пароль для следующей попытки
                check_password(category, force_new=True)
                return {"text": "", "passed": True}

            # ── ШАГ 1: Общий BERT (опасный / безопасный) ────────────────────
            general_result = self.general_bert.classify(text)

            if general_result == 0:
                # Безопасный → LLM
                return {
                    "text": self._call_llm(text, category, current_password),
                    "passed": False,
                }

            # ── ШАГ 2: Запрос опасный → проверяем категорию ─────────────────
            cat_classifier = self._get_category_classifier(category)
            is_target_category = cat_classifier.classify(text)

            if not is_target_category:
                return {
                    "text": (
                        f"Категория вашего запроса не соответствует выбранной "
                        f"(вы выбрали категорию {category}). "
                        f"Попробуйте выбрать другую категорию или переформулировать запрос."
                    ),
                    "passed": False,
                }

            # ── ШАГ 3: TF-IDF классификатор (фиксированный уровень) ─────────
            level_classifier = self.level_classifiers[GAME_DIFFICULTY]
            level_result = level_classifier.classify(text)

            if level_result == 1:
                return {
                    "text": (
                        f"Модель ({level_classifier.get_name()}) поняла, что запрос опасный "
                        f"и его нужно усилить. Попробуйте обойти защиту этого уровня."
                    ),
                    "passed": False,
                }

            # ── ШАГ 4: TF-IDF пропустил → LLM ───────────────────────────────
            return {
                "text": self._call_llm(text, category, current_password),
                "passed": False,
            }

        except Exception as e:
            return {
                "text": f"Сервис временно недоступен: {str(e)}",
                "passed": False,
            }

    def _call_llm(self, query: str, category: int, current_password: str) -> str:
        """Вызывает LLM, явно передавая текущий пароль."""
        try:
            return self.llm_client.get_response(
                user_prompt=query,
                ghost_id=category,
                current_password=current_password,
            )
        except Exception as e:
            return f"Ошибка при обращении к LLM: {str(e)}"