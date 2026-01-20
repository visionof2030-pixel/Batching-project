import datetime
from fastapi import HTTPException
from database import SessionLocal, AccessKey
from security import hash_key

def validate_time_key(access_key: str):
    db = SessionLocal()
    key_hash = hash_key(access_key)

    key = db.query(AccessKey).filter_by(key_hash=key_hash).first()

    if not key or not key.is_active:
        raise HTTPException(status_code=403, detail="Invalid access key")

    if datetime.datetime.utcnow() > key.expires_at:
        raise HTTPException(status_code=403, detail="Access key expired")

    return key