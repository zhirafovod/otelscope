"""
Entry point for the OTelScope CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from contextlib import suppress

from .server import TelemetryServer, TelemetryServerConfig
from .store import TraceStore
from .ui import OTelScopeUI

logger = logging.getLogger(__name__)


async def _run_async(args) -> None:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    store = TraceStore(retention_traces=args.retention)
    config = TelemetryServerConfig(grpc_endpoint=args.grpc_endpoint, http_endpoint=args.http_endpoint)
    server = TelemetryServer(store, config)
    ui = OTelScopeUI(store, config, on_exit=stop_event.set)

    def _handle_signal() -> None:
        logger.info("Received shutdown signal, stopping...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handle_signal)

    await server.start()

    ui_task = asyncio.create_task(ui.run())
    await stop_event.wait()

    await ui.stop()
    with suppress(asyncio.CancelledError):
        await ui_task

    await server.stop()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OTelScope: interactive OpenTelemetry telemetry explorer.")
    parser.add_argument(
        "--grpc-endpoint",
        default="127.0.0.1:4317",
        help="Address for OTLP gRPC ingestion (default: %(default)s)",
    )
    parser.add_argument(
        "--http-endpoint",
        default="127.0.0.1:4318",
        help="Address for OTLP HTTP ingestion (default: %(default)s)",
    )
    parser.add_argument(
        "--retention",
        type=int,
        default=100,
        help="Maximum number of traces retained in memory (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for OTelScope (default: %(default)s)",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(list(argv))
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    run()
