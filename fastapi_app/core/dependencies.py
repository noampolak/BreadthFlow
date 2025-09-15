from core.config import settings
from core.database import get_db
from fastapi import Depends
from shared.websocket import WebSocketManager
from sqlalchemy.orm import Session


def get_settings():
    return settings


def get_db_session():
    yield from get_db()


websocket_manager = WebSocketManager()
