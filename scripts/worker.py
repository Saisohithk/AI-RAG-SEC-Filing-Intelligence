"""
scripts/worker.py — RQ worker process for ingestion jobs.

Start alongside the API when Redis is configured:
  python scripts/worker.py

Or via Docker Compose (see docker-compose.yml):
  docker compose up worker
"""

import logging
import sys
from pathlib import Path

# Ensure src/ is importable when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from rq import Queue, Worker

from src.core.config import settings
from src.core.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    if not settings.REDIS_URL:
        logger.error("REDIS_URL not set — worker cannot start without Redis")
        sys.exit(1)

    conn = redis.from_url(settings.REDIS_URL)
    queue = Queue("ingestion", connection=conn)
    worker = Worker([queue], connection=conn)

    logger.info(f"RQ worker starting — listening on queue 'ingestion' at {settings.REDIS_URL}")
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
