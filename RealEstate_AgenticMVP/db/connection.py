from google.cloud.sql.connector import AsyncConnector
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from config.settings import settings

# Cloud SQL connector handles IAM auth + SSL automatically
connector = AsyncConnector()


async def get_connection():
    """Return an asyncpg connection via Cloud SQL connector."""
    return await connector.connect_async(
        settings.cloud_sql_instance,
        "asyncpg",
        user=settings.db_user,
        password=settings.db_password,
        db=settings.db_name,
    )


# SQLAlchemy async engine — used by all tools for queries
engine = create_async_engine(
    "postgresql+asyncpg://",
    async_creator=get_connection,
    echo=False,
)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncSession:
    """Dependency — yields a DB session and closes it after use."""
    async with AsyncSessionLocal() as session:
        yield session
