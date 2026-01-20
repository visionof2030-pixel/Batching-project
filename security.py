import hashlib

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()