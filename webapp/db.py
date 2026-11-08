"""Task 5 - database trail. Every prediction becomes one row that answers:
what was decided, when, by which model, and how confidently.

DATABASE_URL picks the database:
  postgresql+psycopg://user:pass@db:5432/visionguard   (Docker Compose / cloud)
  sqlite:///visionguard.db                             (quick local runs, the default)
"""
import time
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True,
                                                 default=lambda: datetime.now(timezone.utc))
    filename: Mapped[str] = mapped_column(String(200))          # sanitized name, never the file itself
    label: Mapped[str] = mapped_column(String(10))               # "ai" / "real"
    confidence: Mapped[float] = mapped_column(Float)
    p_ai: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(20))
    inference_source: Mapped[str] = mapped_column(String(200))  # "local" or the API address
    latency_ms: Mapped[float] = mapped_column(Float)


class Database:
    def __init__(self, url: str, retries: int = 10):
        self.url = url
        self.engine = create_engine(url, pool_pre_ping=True, future=True)
        for attempt in range(1, retries + 1):     # the DB container may still be starting
            try:
                Base.metadata.create_all(self.engine)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == retries:
                    raise
                print(f"[db] not ready ({type(exc).__name__}), retry {attempt}/{retries} in 2s")
                time.sleep(2)

    def log(self, pred, filename: str, source: str) -> PredictionLog:
        row = PredictionLog(filename=filename, label=pred.label, confidence=pred.confidence,
                            p_ai=pred.p_ai, threshold=pred.threshold, model_name=pred.model_name,
                            model_version=str(pred.model_version), inference_source=source,
                            latency_ms=pred.latency_ms)
        with Session(self.engine, expire_on_commit=False) as s:
            s.add(row)
            s.commit()
        return row

    def recent(self, limit: int = 100):
        with Session(self.engine) as s:
            return list(s.scalars(select(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit)))

    def stats(self) -> dict:
        with Session(self.engine) as s:
            total = s.scalar(select(func.count(PredictionLog.id))) or 0
            ai = s.scalar(select(func.count(PredictionLog.id)).where(PredictionLog.label == "ai")) or 0
        return {"total": total, "ai": ai, "real": total - ai}

    def ping(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False
