"""
Database engine/session setup.

Production: set DATABASE_URL to a Postgres DSN, e.g.
    postgresql+psycopg2://user:pass@host:5432/cardpilot

Local dev/testing (no Postgres server required): leave DATABASE_URL unset
and this falls back to a local SQLite file. The ORM models, entitlement
engine, and API code never change between the two -- only this one line
does, which is the whole point of going through SQLAlchemy instead of
hand-rolled SQL.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./cardpilot_dev.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
