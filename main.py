"""
FastAPI backend для сервиса защиты от промпт-инъекций.

API:
  POST /predict  — игровой эндпоинт: level + text → text + passed
  GET  /health   — health-check
  GET  /         — статический фронтенд
"""

import os
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from pipeline import ProcessingPipeline

# ── Приложение ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Prompt Injection Defense Service",
    description="Сервис для обучения защите от промпт-инъекций LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Пайплайн ─────────────────────────────────────────────────────────────────

pipeline: Optional[ProcessingPipeline] = None


@app.on_event("startup")
async def startup_event():
    global pipeline
    print("[*] Инициализация пайплайна...")
    try:
        pipeline = ProcessingPipeline()
        print("[+] Пайплайн успешно инициализирован!")
    except Exception as e:
        print(f"[!] Ошибка инициализации пайплайна: {e}")
        raise


# ── Модели ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    level: Literal["tony_stark", "subnautica", "cyberpunk"] = Field(
        ...,
        description="Игровой уровень: tony_stark (кат. 3), subnautica (кат. 4), cyberpunk (кат. 5)",
    )
    text: str = Field(..., min_length=1, description="Текст пользователя")


class PredictResponse(BaseModel):
    text: str = Field(..., description="Ответ сервиса (пустая строка, если пароль угадан)")
    passed: bool = Field(..., description="True — пользователь угадал пароль и прошёл уровень")


# ── Эндпоинты ────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(data: PredictRequest):
    """
    Игровой эндпоинт.

    Принимает level и text, возвращает text и passed.
    Пайплайн:
      1. Инициализирует/берёт пароль для категории уровня.
      2. Если text совпал с паролем → passed=True, text="", пароль меняется.
      3. Иначе — прогоняет через BERT → категорию → TF-IDF → LLM с текущим паролем.
      4. Всегда возвращает ответ, даже при внутренней ошибке.
    """
    if pipeline is None:
        return PredictResponse(
            text="Сервис временно недоступен. Пайплайн не инициализирован.",
            passed=False,
        )

    try:
        result = pipeline.process(text=data.text, level=data.level)
        return PredictResponse(text=result["text"], passed=result["passed"])
    except Exception as e:
        return PredictResponse(
            text=f"Произошла непредвиденная ошибка: {str(e)}",
            passed=False,
        )


@app.get("/health")
async def health_check():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


# ── Фронтенд ─────────────────────────────────────────────────────────────────

static_dir = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not found")


# ── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)