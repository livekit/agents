"""
Tiny aiohttp server so you can update ignored words at runtime.
This is optional bonus; included for completeness.
"""
import asyncio
from aiohttp import web
import json
from .handler import InterruptHandler

routes = web.RouteTableDef()

# We'll create a singleton handler in the module for the small server
_global_handler: InterruptHandler = None

def create_app(handler: InterruptHandler):
    global _global_handler
    _global_handler = handler
    app = web.Application()
    app.add_routes(routes)
    return app

@routes.get("/ignored_words")
async def get_ignored(request):
    return web.json_response({"ignored_words": _global_handler.ignored_words})

@routes.post("/ignored_words")
async def update_ignored(request):
    data = await request.json()
    new = data.get("ignored_words")
    if not isinstance(new, list):
        return web.json_response({"error": "expected ignored_words as list"}, status=400)
    await _global_handler.update_ignored_words(new)
    return web.json_response({"status": "ok", "ignored_words": _global_handler.ignored_words})
