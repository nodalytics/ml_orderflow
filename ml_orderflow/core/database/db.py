import asyncio
from typing import Dict

from skincare_ai.utils.config import settings
from skincare_ai.core.database import DataBaseToUse


class DataBase:
    def __init__(self):
        self._db = DataBaseToUse['mongodb']

    async def get_database_config_config_details(self):
        return self._db

    async def update_single_db_record(self, record: Dict):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._db.update_single_db_record(record))

    async def update_multiple_db_record(self, record: Dict):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._db.update_multiple_db_record(record))

    async def fetch_single_db_record(self, unique_id: str):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._db.fetch_single_db_record(unique_id))

    async def fetch_multiple_db_record(self, unique_id: str):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._db.fetch_multiple_db_record(unique_id))

    async def insert_single_db_record(self, record: Dict):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._db.insert_single_db_record(record))
        loop.close()

    async def insert_multiple_db_record(self, record: Dict):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._db.insert_multiple_db_record(record))
        loop.close()
