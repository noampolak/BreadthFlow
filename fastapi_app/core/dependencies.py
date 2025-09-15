from fastapi import Depends
from sqlalchemy.orm import Session
from core.database import get_db
from core.config import settings
from shared.websocket import WebSocketManager


def get_settings():
    return settings


def get_db_session():
    yield from get_db()


websocket_manager = WebSocketManager()
