import secrets, datetime
from database import SessionLocal, AccessKey
from security import hash_key

db = SessionLocal()

plain_key = secrets.token_hex(16)

key = AccessKey(
    key_hash=hash_key(plain_key),
    expires_at=datetime.datetime.utcnow() + datetime.timedelta(days=7),  # غيّرها متى شئت
    is_active=True
)

db.add(key)
db.commit()

print("ACCESS KEY:", plain_key)
print("EXPIRES AT:", key.expires_at)