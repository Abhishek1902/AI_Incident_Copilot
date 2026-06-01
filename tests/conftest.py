# Set required env vars before any app module is imported.
# pydantic-settings reads them at Settings() construction time (module import),
# so these must be set here, at the top of conftest, before any test file import.
import os
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5433/testdb")
