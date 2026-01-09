"""
Dependency injection for FastAPI
"""
from app.database.connection import get_db


def get_database():
    """Dependency for database connection"""
    return get_db()
