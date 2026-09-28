"""Vercel entrypoint for the FastAPI backend."""

from backend.app.main import app

__all__ = ["app"]