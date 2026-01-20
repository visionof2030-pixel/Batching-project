from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

DATABASE_URL = "sqlite:///licenses.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class LicenseKey(Base):
    __tablename__ = "license_keys"

    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, index=True)
    expires_at = Column(DateTime)
    max_requests = Column(Integer)
    used_requests = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

    def is_valid(self):
        return (
            self.is_active
            and self.used_requests < self.max_requests
            and datetime.datetime.utcnow() < self.expires_at
        )

Base.metadata.create_all(engine)