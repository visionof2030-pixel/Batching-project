import uuid
import datetime
from database import SessionLocal, LicenseKey

def create_license(days=30, max_requests=1000):
    db = SessionLocal()

    key_value = str(uuid.uuid4()).replace("-", "").upper()
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=days)

    license_key = LicenseKey(
        key=key_value,
        expires_at=expires_at,
        max_requests=max_requests,
        used_requests=0,
        is_active=True
    )

    db.add(license_key)
    db.commit()
    db.close()

    return key_value, expires_at, max_requests

if __name__ == "__main__":
    key, exp, limit = create_license(
        days=30,          # مدة الاشتراك
        max_requests=1000 # عدد الطلبات المسموح
    )

    print("✅ تم إنشاء مفتاح جديد")
    print("🔑 المفتاح:", key)
    print("📅 ينتهي في:", exp)
    print("📊 الحد الأقصى:", limit)