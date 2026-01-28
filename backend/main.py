import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models.db import connect_to_mongo, close_mongo_connection
from api.auth import router as auth_router
from api.admin import router as admin_router
from api.doctor import router as doctor_router
from settings import settings


app = FastAPI(
    title="Medical Guideline Study Platform",
    version="0.1.0",
    description=(
        "Study & reference only. Retrieval-first, extractive outputs. "
        "Not a chatbot, not a medical decision system."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    await connect_to_mongo()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await close_mongo_connection()


app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
app.include_router(doctor_router, prefix="/api/doctor", tags=["doctor"])

# Serves extracted images at /uploads/...
os.makedirs(settings.upload_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")
