from __future__ import annotations

import asyncio
import datetime as dt
import json
import subprocess as sp
import time
import urllib.parse
from dataclasses import dataclass, InitVar
from time import sleep

import aiohttp
import pytest as pytest
import requests
from typing import Any


def wait_for_ch(hostname: str, port: int) -> None:
    for _ in range(10):
        print(f"Waiting for ClickHouse on {hostname}:{port} to start")
        sleep(1)
        if check_ch(hostname, port):
            print(f"ClickHouse started on {hostname}:{port}")
            return
    raise Exception(f"ClickHouse has not started on {hostname}:{port}")


def check_ch(hostname: str, port: int):
    try:
        url = f"http://{hostname}:{port}?query={urllib.parse.quote('SELECT 42 AS dummy FORMAT JSON')}"
        return requests.get(url).json()["data"] == [{"dummy": 42}]
    except Exception:
        return False


def client_factory(hostname: str, port: int) -> ClientFactory:
    if not check_ch(hostname, port):
        command = f"podman run -d -p {port}:8123 docker.io/clickhouse/clickhouse-server".split(" ")
        if sp.run(command).returncode != 0:
            raise Exception(f"Command failed: {command}")
        wait_for_ch(hostname, port)
    return ClientFactory(hostname, port)


@pytest.fixture
def source_factory() -> ClientFactory:
    return client_factory("localhost", 8128)


@pytest.fixture
def target_factory() -> ClientFactory:
    return client_factory("localhost", 8129)


@dataclass
class ClientFactory:
    hostname: str
    port: int

    def create(self, session):
        return AsyncClient(hostname=self.hostname, port=self.port, session=session)

    def __call__(self) -> ClientContext:
        return ClientContext(self.hostname, self.port, aiohttp.ClientSession())


@dataclass
class ClientContext:
    hostname: str
    port: int
    session: Any

    async def __aenter__(self):
        return AsyncClient(hostname=self.hostname, port=self.port, session=await self.session.__aenter__())

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.__aexit__(exc_type, exc_val, exc_tb)


async def populate_table(client: AsyncClient, num_dates: int = 40, rows_per_date: int = 1000, num_dims: int = 100):
    await client.post(f"DROP DATABASE IF EXISTS example")
    await client.post(f"CREATE DATABASE example")

    dims = ["col_%d" % i for i in range(num_dims)]

    await client.post(f"DROP TABLE IF EXISTS example._random")
    await client.post(f"DROP TABLE IF EXISTS example.t1")

    await client.post(f"""
        CREATE TABLE example._random 
        ({', '.join(f'{di} String' for di in dims)}, value DOUBLE) 
        ENGINE=GenerateRandom(123123, 10) """)

    await client.post(f"""
        CREATE TABLE example.t1
        (date String, {', '.join(f'{di} String' for di in dims)}, value DOUBLE)
        ENGINE MergeTree PRIMARY KEY ({', '.join(dims)}) PARTITION by date """)

    for i in range(num_dates):
        print(f"Inserting {rows_per_date} rows ({i + 1}/{num_dates})")
        date = dt.date(2022, 1, 1) + dt.timedelta(days=i)
        await client.post(
            f"INSERT INTO example.t1 SELECT '{date}', r.* FROM example._random AS r LIMIT {rows_per_date}")

    await client.post(f"DROP TABLE example._random")


@dataclass
class Copier:
    source: AsyncClient
    target: AsyncClient
    database: str
    num_readers: InitVar[int]
    num_writers: InitVar[int]
    buffer: InitVar[int]

    def __post_init__(self, num_readers, num_writers, buffer):
        self.read_semaphore = asyncio.Semaphore(num_readers)
        self.write_semaphore = asyncio.Semaphore(num_writers)
        self.buffer_semaphore = asyncio.Semaphore(buffer)
        self.num_pending = 0
        self.num_queued = 0
        self.num_written = 0

    def update(self, num_pending=0, num_queued=0, num_written=0):
        self.num_pending += num_pending
        self.num_queued += num_queued
        self.num_written += num_written
        print(f"{self.num_pending}/{self.num_queued}/{self.num_written}")

    async def read_partition(self, table: str, partition_id: str):
        async with self.read_semaphore:
            await self.buffer_semaphore.acquire()
            query = f"SELECT * FROM `{self.database}`.`{table}` WHERE _partition_id = '{partition_id}' FORMAT Native"
            return await self.source.get(query)

    async def get_tables(self):
        result = await self.source.get_json(
            f"SELECT table from system.tables WHERE database = '{self.database}'")
        return [row["table"] for row in result]

    async def get_partition_ids(self, table):
        result = await self.source.get_json(f"""
            SELECT DISTINCT partition_id FROM system.parts 
            WHERE database = '{self.database}'
            AND table = '{table}' ORDER BY partition """)

        return [row["partition_id"] for row in result]

    async def get_total_bytes(self) -> int:
        result = await self.source.get_json(f"""
            SELECT SUM(total_bytes) AS total_bytes FROM system.tables 
            WHERE database = '{self.database}' """)
        return int(result[0]["total_bytes"])

    async def get_ddl(self, table) -> str:
        result = await self.source.get_json(f"SHOW TABLE `{self.database}`.`{table}`")
        return result[0]["statement"]

    async def write_partition(self, table: str, data: bytes):
        async with self.write_semaphore:
            self.buffer_semaphore.release()
            await self.target.post(f"INSERT INTO `{self.database}`.`{table}` FORMAT Native", data)

    async def copy_partition(self, table: str, partition_id: str):
        data = await self.read_partition(table, partition_id)
        self.update(num_pending=-1, num_queued=+1)
        await self.write_partition(table, data)
        self.update(num_queued=-1, num_written=+1)

    async def copy(self):
        start_time = time.time()

        tables = await self.get_tables()
        partitions = [(ti, pi) for ti in tables for pi in await self.get_partition_ids(ti)]

        total_bytes = await self.get_total_bytes()

        self.update(num_pending=len(partitions))

        await self.target.post(f"DROP DATABASE IF EXISTS {self.database}")
        await self.target.post(f"CREATE DATABASE {self.database}")
        for ti in tables:
            await self.target.post(await self.get_ddl(ti))

        tasks = [asyncio.create_task(self.copy_partition(ti, pi)) for ti, pi in partitions]
        await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        print(
            f"Copied {total_bytes / 1024 / 1024:.0f}MB in {total_time:.3f}s: {total_bytes / 1024 / 1024 / total_time:.1f}MB/s")


@dataclass(frozen=True)
class AsyncClient:
    session: Any
    hostname: str
    port: int

    def url(self, query):
        return f"http://{self.hostname}:{self.port}?query={urllib.parse.quote(query)}"

    async def get(self, query):
        async with self.session.get(self.url(query)) as response:
            if response.status != 200:
                raise Exception(f"Error {response.status} {response.text}")
            return await response.content.read()

    async def post(self, query, data=None):
        async with self.session.post(self.url(query), data=data) as response:
            if response.status != 200:
                raise Exception(f"Error {response.status} {response.text}")

    async def get_json(self, query):
        result = await self.get(f"{query} FORMAT JSON")
        return json.loads(result)["data"]


def test_populate(source_factory) -> None:
    async def populate():
        async with source_factory() as client:
            await populate_table(client, num_dates=40, rows_per_date=3000, num_dims=100)

    asyncio.run(populate())


def test_async_copy(source_factory, target_factory):
    print("\n")

    async def copy():
        async with source_factory() as source, target_factory() as target:
            copier = Copier(source=source, target=target, database="example", num_readers=4,
                            num_writers=8, buffer=12)
            await copier.copy()

    asyncio.run(copy())
