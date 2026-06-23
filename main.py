"""
FastAPI backend для сервиса защиты от промпт-инъекций.

API:
    POST /process — обработка запроса через пайплайн классификаторов
    GET  /      — отдаёт статический HTML (фронтенд)
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from pipeline import ProcessingPipeline


# ============================================================
# Инициализация FastAPI приложения
# ============================================================
app = FastAPI(
    title="Prompt Injection Defense Service",
    description="Сервис для обучения защите от промпт-инъекций LLM",
    version="1.0.0"
)

# CORS (для разработки — разрешаем всё)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Пайплайн обработки (инициализируется при старте)
# ============================================================
pipeline: Optional[ProcessingPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Инициализация при старте сервера."""
    global pipeline
    print("[*] Инициализация пайплайна...")
    try:
        pipeline = ProcessingPipeline()
        print("[+] Пайплайн успешно инициализирован!")
    except Exception as e:
        print(f"[!] Ошибка инициализации пайплайна: {e}")
        raise


# ============================================================
# Pydantic модели запросов/ответов
# ============================================================
class ProcessRequest(BaseModel):
    """Запрос на обработку через пайплайн."""
    query: str = Field(..., min_length=1, description="Текст запроса пользователя")
    category: int = Field(..., ge=3, le=5, description="Целевая категория (3, 4 или 5)")
    difficulty: int = Field(..., ge=1, le=3, description="Уровень сложности: 1-TF-IDF, 2-LSTM, 3-BERT")


class ProcessResponse(BaseModel):
    """Ответ от пайплайна."""
    response: str = Field(..., description="Текстовый ответ от сервиса")
    status: str = Field("ok", description="Статус обработки")


# ============================================================
# API Endpoints
# ============================================================
@app.post("/process", response_model=ProcessResponse)
async def process_request(data: ProcessRequest):
    """
    Обрабатывает запрос пользователя через цепочку классификаторов:
    1. Общий BERT (опасный/безопасный)
    2. Категориальный BERT (проверка целевой категории)
    3. Классификатор уровня (TF-IDF/LSTM/BERT)
    4. LLM (генерация ответа)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Пайплайн не инициализирован")

    try:
        result = pipeline.process(
            query=data.query,
            category=data.category,
            difficulty=data.difficulty
        )
        return ProcessResponse(response=result, status="ok")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса."""
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None
    }


# ============================================================
# Статический фронтенд
# ============================================================
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_frontend():
    """Отдаёт главную страницу (фронтенд)."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not found")


# ============================================================
# Точка входа
# ============================================================
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)
