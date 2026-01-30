from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
import bcrypt

from models.db import get_db
from models.user import LoginRequest, TokenResponse, UserCreate, UserPublic
from settings import settings


router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password[:72].encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hash_value: str) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(password[:72].encode(), hash_value.encode())
    except Exception:
        return False


def _create_token(user_id: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_alg)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        db = get_db()
        user = await db["users"].find_one({"_id": user_id})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user
    except JWTError as e:
        raise HTTPException(status_code=401, detail="Invalid token") from e


def require_role(role: str):
    async def _dep(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") != role:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

    return _dep


def require_roles(*roles: str):
    """Allow any of the given roles (e.g. admin or doctor for search)."""
    async def _dep(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

    return _dep


@router.post("/signup", response_model=UserPublic)
async def signup(body: UserCreate):
    db = get_db()
    existing = await db["users"].find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = f"user_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    # Hash password using bcrypt
    hashed = hash_password(body.password)
    doc = {"_id": user_id, "email": body.email, "password_hash": hashed, "role": body.role}
    await db["users"].insert_one(doc)
    return UserPublic(id=user_id, email=body.email, role=body.role)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    db = get_db()
    user = await db["users"].find_one({"email": body.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    password_hash = user.get("password_hash", "")
    is_valid = verify_password(body.password, password_hash)
    
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = _create_token(user["_id"], user["role"])
    return TokenResponse(access_token=token)

