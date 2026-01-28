from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from settings import settings

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_to_mongo() -> None:
    global _client, _db
    if _client is not None:
        return
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _db = _client[settings.mongodb_db]

    # Basic indexes (idempotent)
    await _db["users"].create_index("email", unique=True)
    await _db["documents"].create_index("pdf_name")
    await _db["chunks"].create_index([("doc_id", 1), ("page", 1)])


async def close_mongo_connection() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB not connected yet")
    return _db

