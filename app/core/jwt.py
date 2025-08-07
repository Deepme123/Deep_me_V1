# app/core/jwt.py

from jose import jwt, JWTError
from datetime import datetime, timedelta
import os

SECRET_KEY = settings.JWT_SECRET_KEY 
ALGORITHM = "HS256"
EXPIRE_MINUTES = 60  # 1시간

def create_access_token(user_id: str, expires_delta: timedelta = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=EXPIRE_MINUTES))
    payload = {
        "sub": user_id,
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
