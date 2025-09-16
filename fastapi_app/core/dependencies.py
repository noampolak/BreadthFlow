from fastapi_app.core.config import settings
from fastapi_app.core.database import get_db
from fastapi import Depends
from fastapi_app.shared.websocket import WebSocketManager
from sqlalchemy.orm import Session


def get_settings():
    return settings


def get_db_session():
    yield from get_db()


websocket_manager = WebSocketManager()
