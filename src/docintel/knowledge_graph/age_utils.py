"""Utility helpers for configuring PostgreSQL AGE sessions."""

from __future__ import annotations

from ..config import AgeGraphSettings


def configure_age_session(cursor, settings: AgeGraphSettings, *, ensure_graph: bool = True) -> None:
    """Initialise an AGE-capable session using a synchronous cursor."""

    if not settings.enabled:
        return
    if settings.load_extension:
        cursor.execute("LOAD 'age';")
    if settings.search_path:
        cursor.execute(f"SET search_path = {settings.search_path}")
    if not ensure_graph:
        return
    cursor.execute(
        "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s LIMIT 1",
        (settings.graph_name,),
    )
    exists = cursor.fetchone()
    if not exists:
        cursor.execute("SELECT create_graph(%s)", (settings.graph_name,))


async def configure_age_session_async(cursor, settings: AgeGraphSettings, *, ensure_graph: bool = True) -> None:
    """Initialise an AGE-capable session using an asynchronous cursor."""

    if not settings.enabled:
        return
    if settings.load_extension:
        await cursor.execute("LOAD 'age';")
    if settings.search_path:
        await cursor.execute(f"SET search_path = {settings.search_path}")
    if not ensure_graph:
        return
    await cursor.execute(
        "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s LIMIT 1",
        (settings.graph_name,),
    )
    exists = await cursor.fetchone()
    if not exists:
        await cursor.execute("SELECT create_graph(%s)", (settings.graph_name,))
