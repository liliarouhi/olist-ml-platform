import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DatabaseConfig:

    dbname: str = os.getenv("DB_NAME") or "olist"
    user: str = os.getenv("DB_USER") or os.getenv("USER") or "postgres"
    password: str = os.getenv("DB_PASSWORD") or ""
    host: str = os.getenv("DB_HOST") or "localhost"
    port: int = int(os.getenv("DB_PORT") or "5432")

    def conninfo(self) -> str:
        """Return a libpq connection string."""
        parts = [f"dbname={self.dbname}", f"user={self.user}",
                 f"host={self.host}", f"port={self.port}"]
        if self.password:
            parts.append(f"password={self.password}")
        return " ".join(parts)