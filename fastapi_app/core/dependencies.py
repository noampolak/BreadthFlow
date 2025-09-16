from fastapi import Depends
from sqlalchemy.orm import Session

from fastapi_app.core.config import settings
from fastapi_app.core.database import get_db
from fastapi_app.shared.websocket import WebSocketManager


def get_settings():
    return settings


def get_db_session():
    yield from get_db()


websocket_manager = WebSocketManager()
