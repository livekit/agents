"""
REST API Handler for Dynamic Filler Updates

This module provides a simple HTTP endpoint to dynamically update filler words
at runtime without restarting the agent.

Usage:
    POST /update_filler
    Content-Type: application/json
    {
        "add": ["arey", "yaar"],
        "remove": ["okay", "ok"]
    }

Author: Raghav (LiveKit Intern Assessment - Bonus Feature)
Date: November 19, 2025
"""

from __future__ import annotations

import asyncio
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .filler_filter import FillerFilter

from ..log import logger


class FillerUpdateHandler(BaseHTTPRequestHandler):
    """HTTP request handler for dynamic filler word updates."""

    # Class variable to hold the filter instance
    filler_filter: FillerFilter | None = None

    def do_POST(self) -> None:
        """Handle POST requests to /update_filler endpoint."""

        if self.path != "/update_filler":
            self.send_error(404, "Endpoint not found. Use /update_filler")
            return

        if not self.filler_filter:
            self.send_error(500, "Filler filter not initialized")
            return

        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            # Extract add/remove lists
            add = data.get("add", [])
            remove = data.get("remove", [])

            # Validate input
            if not isinstance(add, list) or not isinstance(remove, list):
                self.send_error(400, "Invalid request format. Expected {\"add\": [], \"remove\": []}")
                return

            # Update filter (run async operation in sync context)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.filler_filter.update_fillers_dynamic(add=add, remove=remove)
            )
            loop.close()

            # Send success response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            response = {
                "status": "success",
                "message": "Filler words updated successfully",
                "result": result,
            }

            self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))

            logger.info(
                "[API] Filler words updated via REST API",
                extra={
                    "added": result["added"],
                    "removed": result["removed"],
                    "total_fillers": len(result["current_fillers"]),
                },
            )

        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON in request body")
        except Exception as e:
            self.send_error(500, f"Internal server error: {str(e)}")
            logger.error(f"[API] Error updating fillers: {e}")

    def do_GET(self) -> None:
        """Handle GET requests to retrieve current filler words."""

        if self.path == "/fillers":
            if not self.filler_filter:
                self.send_error(500, "Filler filter not initialized")
                return

            try:
                # Get current fillers
                current_fillers = self.filler_filter.get_ignored_words()
                current_language = self.filler_filter.get_current_language()
                available_languages = self.filler_filter.get_available_languages()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                response = {
                    "status": "success",
                    "current_fillers": current_fillers,
                    "current_language": current_language,
                    "available_languages": available_languages,
                    "total_count": len(current_fillers),
                }

                self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))

            except Exception as e:
                self.send_error(500, f"Internal server error: {str(e)}")

        elif self.path == "/":
            # Info endpoint
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            info = {
                "service": "LiveKit Filler Filter API",
                "version": "1.0",
                "endpoints": {
                    "POST /update_filler": "Update filler words",
                    "GET /fillers": "Get current filler words",
                    "POST /switch_language": "Switch language",
                },
                "example_request": {
                    "url": "/update_filler",
                    "method": "POST",
                    "body": {
                        "add": ["arey", "yaar"],
                        "remove": ["okay"]
                    }
                }
            }

            self.wfile.write(json.dumps(info, indent=2).encode("utf-8"))

        else:
            self.send_error(404, "Endpoint not found")

    def log_message(self, format: str, *args) -> None:
        """Override to use LiveKit logger instead of printing to stderr."""
        logger.info(f"[API] {format % args}")


def start_filler_api_server(
    filler_filter: FillerFilter,
    port: int = 8080,
    host: str = "localhost"
) -> HTTPServer:
    """
    Start the HTTP server for dynamic filler updates.

    Args:
        filler_filter: The FillerFilter instance to manage
        port: Port to listen on (default: 8080)
        host: Host to bind to (default: localhost)

    Returns:
        HTTPServer instance (needs to be run in a separate thread/task)

    Example:
        # In your agent setup
        filter = FillerFilter()
        server = start_filler_api_server(filter, port=8080)

        # Run in background thread
        import threading
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()

        # Or run async
        loop.run_in_executor(None, server.serve_forever)
    """
    FillerUpdateHandler.filler_filter = filler_filter

    server = HTTPServer((host, port), FillerUpdateHandler)

    logger.info(
        "[API] Filler update API server started",
        extra={
            "host": host,
            "port": port,
            "endpoints": [
                f"POST http://{host}:{port}/update_filler",
                f"GET http://{host}:{port}/fillers",
            ]
        },
    )

    return server


# Example usage function
async def example_api_usage():
    """
    Example of how to use the API.

    This can be called from your agent code to enable runtime updates.
    """
    import threading

    from .filler_filter import FillerFilter

    # Create filter
    filter = FillerFilter(enable_multi_language=True)

    # Start API server in background
    server = start_filler_api_server(filter, port=8080)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    logger.info("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         FILLER FILTER API SERVER RUNNING                       ║
    ╟────────────────────────────────────────────────────────────────╢
    ║  Test the API with curl:                                       ║
    ║                                                                ║
    ║  # Get current fillers                                         ║
    ║  curl http://localhost:8080/fillers                            ║
    ║                                                                ║
    ║  # Add new fillers                                             ║
    ║  curl -X POST http://localhost:8080/update_filler \\            ║
    ║    -H "Content-Type: application/json" \\                      ║
    ║    -d '{"add": ["arey", "yaar"]}'                              ║
    ║                                                                ║
    ║  # Remove fillers                                              ║
    ║  curl -X POST http://localhost:8080/update_filler \\            ║
    ║    -H "Content-Type: application/json" \\                      ║
    ║    -d '{"remove": ["okay", "ok"]}'                             ║
    ║                                                                ║
    ║  # Both add and remove                                         ║
    ║  curl -X POST http://localhost:8080/update_filler \\            ║
    ║    -H "Content-Type: application/json" \\                      ║
    ║    -d '{"add": ["haan"], "remove": ["umm"]}'                   ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    return server
