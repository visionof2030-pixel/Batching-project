import uuid, datetime
from database import SessionLocal, LicenseKey

db = SessionLocal()

def create_key(days=30, max_requests=500):
    key = str(uuid.uuid4()).replace("-", "").upper()
    expires = datetime.datetime.utcnow() + datetime.timedelta(days=days)

    lk = LicenseKey(
        key=key,
        expires_at=expires,
        max_requests=max_requests
    )
    db.add(lk)
    db.commit()
    return key

if __name__ == "__main__":
    new_key = create_key(days=30, max_requests=1000)
    print("NEW LICENSE KEY:", new_key)