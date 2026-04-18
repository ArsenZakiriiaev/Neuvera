from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = Path(__file__).resolve().parent.parent / "neuvera.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class TestSession(Base):
    __tablename__ = "test_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, default="demo-user", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    kind = Column(String, default="session")  # voice|tremor|movement|session
    status = Column(String, default="ok")
    overall_score = Column(Float, nullable=True)
    overall_confidence = Column(Float, nullable=True)
    voice_score = Column(Float, nullable=True)
    tremor_score = Column(Float, nullable=True)
    movement_score = Column(Float, nullable=True)
    summary = Column(String, nullable=True)
    payload = Column(JSON, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def save_result(kind: str, payload: dict[str, Any], user_id: str = "demo-user") -> int:
    init_db()
    with SessionLocal() as s:
        row = TestSession(
            user_id=user_id,
            kind=kind,
            status=payload.get("status", "ok"),
            overall_score=payload.get("overall_score") or payload.get("score"),
            overall_confidence=payload.get("overall_confidence") or payload.get("confidence"),
            voice_score=(payload.get("voice") or {}).get("score") if kind == "session" else (payload.get("score") if kind == "voice" else None),
            tremor_score=(payload.get("tremor") or {}).get("score") if kind == "session" else (payload.get("score") if kind == "tremor" else None),
            movement_score=(payload.get("movement") or {}).get("score") if kind == "session" else (payload.get("score") if kind == "movement" else None),
            summary=payload.get("summary"),
            payload=payload,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row.id


def list_history(user_id: str = "demo-user", limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with SessionLocal() as s:
        rows = (
            s.query(TestSession)
            .filter(TestSession.user_id == user_id)
            .order_by(TestSession.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() + "Z",
                "kind": r.kind,
                "status": r.status,
                "overall_score": r.overall_score,
                "voice_score": r.voice_score,
                "tremor_score": r.tremor_score,
                "movement_score": r.movement_score,
                "summary": r.summary,
            }
            for r in rows
        ]


def get_result(session_id: int) -> dict[str, Any] | None:
    init_db()
    with SessionLocal() as s:
        r = s.get(TestSession, session_id)
        if not r:
            return None
        return {
            "id": r.id,
            "created_at": r.created_at.isoformat() + "Z",
            "kind": r.kind,
            "payload": r.payload,
        }
