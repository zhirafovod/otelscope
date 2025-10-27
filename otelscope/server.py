"""
OTLP ingestion servers for the OTelScope experience.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import aiohttp
from aiohttp import web
import grpc
from google.protobuf import json_format

from opentelemetry.proto.collector.logs.v1 import logs_service_pb2, logs_service_pb2_grpc
from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2, metrics_service_pb2_grpc
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2, trace_service_pb2_grpc
from opentelemetry.proto.logs.v1 import logs_pb2
from opentelemetry.proto.metrics.v1 import metrics_pb2
from opentelemetry.proto.trace.v1 import trace_pb2
from opentelemetry.proto.common.v1 import common_pb2

from .models import Attachment, SpanNode
from .store import TraceStore

logger = logging.getLogger(__name__)


def _bytes_to_hex(value: bytes) -> str:
    return value.hex()


def _attributes_to_dict(attributes: Sequence[common_pb2.KeyValue]) -> dict:
    result = {}
    for kv in attributes:
        if kv.value.WhichOneof("value") is None:
            continue
        result[kv.key] = _any_value_to_python(kv.value)
    return result


def _any_value_to_python(value: common_pb2.AnyValue):
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "array_value":
        return [_any_value_to_python(v) for v in value.array_value.values]
    if kind == "kvlist_value":
        return {kv.key: _any_value_to_python(kv.value) for kv in value.kvlist_value.values}
    if kind == "bytes_value":
        return value.bytes_value
    return None


def _build_span_node(span: trace_pb2.Span, resource_attrs: dict, scope_attrs: dict) -> SpanNode:
    attributes = _attributes_to_dict(span.attributes)
    attributes.update({f"resource.{k}": v for k, v in resource_attrs.items()})
    attributes.update({f"scope.{k}": v for k, v in scope_attrs.items()})

    status = span.status if span.HasField("status") else None

    return SpanNode(
        trace_id=_bytes_to_hex(span.trace_id),
        span_id=_bytes_to_hex(span.span_id),
        parent_span_id=_bytes_to_hex(span.parent_span_id) if span.parent_span_id else None,
        name=span.name or "(unnamed span)",
        kind=trace_pb2.Span.SpanKind.Name(span.kind),
        start_time_unix_nano=span.start_time_unix_nano,
        end_time_unix_nano=span.end_time_unix_nano,
        attributes=attributes,
        status_message=status.message if status else None,
        status_code=trace_pb2.Status.StatusCode.Name(status.code) if status else None,
        raw_payload=span,
    )


def _build_log_attachment(
    log_record: logs_pb2.LogRecord, resource_attrs: dict, scope_attrs: dict
) -> Attachment:
    trace_id = _bytes_to_hex(log_record.trace_id) if log_record.trace_id else None
    span_id = _bytes_to_hex(log_record.span_id) if log_record.span_id else None
    # LogRecord in the OTLP proto does not define a "name" field. The previous implementation
    # attempted to access log_record.name which raises AttributeError. We derive a stable
    # display name from common attribute conventions or fall back to severity/body.
    attributes = _attributes_to_dict(log_record.attributes)
    summary = (
        log_record.body.string_value
        if log_record.body.HasField("string_value")
        else str(log_record.body)
    )
    name = (
        getattr(log_record, "name", None)  # future-proof if field ever appears
        or attributes.get("event.name")
        or attributes.get("logger.name")
        or attributes.get("log.logger")
        or attributes.get("otel.logger.name")
        or (log_record.severity_text or None)
        or (summary if summary else None)
        or "log"
    )
    attributes.update({f"resource.{k}": v for k, v in resource_attrs.items()})
    attributes.update({f"scope.{k}": v for k, v in scope_attrs.items()})

    detail = {
        "severity_text": log_record.severity_text,
        "severity_number": log_record.severity_number,
        "body": summary,
        "attributes": attributes,
        "time_unix_nano": log_record.time_unix_nano,
    }

    return Attachment(
        attachment_id=f"log-{trace_id}-{span_id}-{log_record.time_unix_nano}",
        span_id=span_id,
        trace_id=trace_id,
        attachment_type="log",
        name=name,
        summary=summary,
        raw_payload=detail,
    )


def _metric_point_summary(metric: metrics_pb2.Metric, point) -> str:
    if isinstance(point, metrics_pb2.NumberDataPoint):
        return f"value={point.as_double if point.HasField('as_double') else point.as_int}"
    if isinstance(point, metrics_pb2.HistogramDataPoint):
        return f"count={point.count} sum={point.sum}"
    if isinstance(point, metrics_pb2.SummaryDataPoint):
        return f"count={point.count} sum={point.sum}"
    return "datapoint"


def _extract_metric_span(point) -> tuple[Optional[str], Optional[str]]:
    if point.exemplars:
        exemplar = point.exemplars[0]
        trace_id = _bytes_to_hex(exemplar.trace_id) if exemplar.trace_id else None
        span_id = _bytes_to_hex(exemplar.span_id) if exemplar.span_id else None
        return trace_id, span_id
    return None, None


def _metric_datapoints(metric: metrics_pb2.Metric) -> Iterable:
    data = metric.WhichOneof("data")
    if data == "gauge":
        return metric.gauge.data_points
    if data == "sum":
        return metric.sum.data_points
    if data == "histogram":
        return metric.histogram.data_points
    if data == "exponential_histogram":
        return metric.exponential_histogram.data_points
    if data == "summary":
        return metric.summary.data_points
    return []


def _build_metric_attachments(
    metric: metrics_pb2.Metric, resource_attrs: dict, scope_attrs: dict
) -> Iterable[Attachment]:
    for point in _metric_datapoints(metric):
        trace_id, span_id = _extract_metric_span(point)

        attributes = {**_attributes_to_dict(point.attributes)}
        attributes.update({f"resource.{k}": v for k, v in resource_attrs.items()})
        attributes.update({f"scope.{k}": v for k, v in scope_attrs.items()})

        summary = _metric_point_summary(metric, point)
        name = metric.name

        detail = {
            "metric": json_format.MessageToDict(metric, preserving_proto_field_name=True),
            "point_attributes": attributes,
        }

        yield Attachment(
            attachment_id=f"metric-{name}-{point.time_unix_nano}",
            span_id=span_id,
            trace_id=trace_id,
            attachment_type="metric",
            name=name if trace_id else f"{name} (unlinked)",
            summary=summary,
            raw_payload=detail,
        )


class _TraceService(trace_service_pb2_grpc.TraceServiceServicer):
    def __init__(self, store: TraceStore):
        self._store = store

    async def Export(self, request: trace_service_pb2.ExportTraceServiceRequest, context) -> trace_service_pb2.ExportTraceServiceResponse:
        for resource_spans in request.resource_spans:
            resource_attrs = _attributes_to_dict(resource_spans.resource.attributes)
            for scope_spans in resource_spans.scope_spans:
                scope_attrs = _attributes_to_dict(scope_spans.scope.attributes)
                for span in scope_spans.spans:
                    span_node = _build_span_node(span, resource_attrs, scope_attrs)
                    await self._store.add_span(span_node)
        return trace_service_pb2.ExportTraceServiceResponse()


class _LogService(logs_service_pb2_grpc.LogsServiceServicer):
    def __init__(self, store: TraceStore):
        self._store = store

    async def Export(self, request: logs_service_pb2.ExportLogsServiceRequest, context) -> logs_service_pb2.ExportLogsServiceResponse:
        for resource_logs in request.resource_logs:
            resource_attrs = _attributes_to_dict(resource_logs.resource.attributes)
            for scope_logs in resource_logs.scope_logs:
                scope_attrs = _attributes_to_dict(scope_logs.scope.attributes)
                for log_record in scope_logs.log_records:
                    attachment = _build_log_attachment(log_record, resource_attrs, scope_attrs)
                    await self._store.add_attachment(attachment)
        return logs_service_pb2.ExportLogsServiceResponse()


class _MetricService(metrics_service_pb2_grpc.MetricsServiceServicer):
    def __init__(self, store: TraceStore):
        self._store = store

    async def Export(self, request: metrics_service_pb2.ExportMetricsServiceRequest, context) -> metrics_service_pb2.ExportMetricsServiceResponse:
        for resource_metrics in request.resource_metrics:
            resource_attrs = _attributes_to_dict(resource_metrics.resource.attributes)
            for scope_metrics in resource_metrics.scope_metrics:
                scope_attrs = _attributes_to_dict(scope_metrics.scope.attributes)
                for metric in scope_metrics.metrics:
                    for attachment in _build_metric_attachments(metric, resource_attrs, scope_attrs):
                        await self._store.add_attachment(attachment)
        return metrics_service_pb2.ExportMetricsServiceResponse()


@dataclass
class TelemetryServerConfig:
    grpc_endpoint: str = "127.0.0.1:4317"
    http_endpoint: str = "127.0.0.1:4318"


class TelemetryServer:
    """Coordinator that runs OTLP gRPC and HTTP servers and forwards data to the store."""

    def __init__(self, store: TraceStore, config: Optional[TelemetryServerConfig] = None) -> None:
        self._store = store
        self._config = config or TelemetryServerConfig()
        self._grpc_server: Optional[grpc.aio.Server] = None
        self._http_runner: Optional[web.AppRunner] = None
        self._http_site: Optional[web.TCPSite] = None
        self._grpc_wait_task: Optional[asyncio.Task] = None
        self._trace_service = _TraceService(self._store)
        self._log_service = _LogService(self._store)
        self._metric_service = _MetricService(self._store)

    async def start(self) -> None:
        await asyncio.gather(self._start_grpc_server(), self._start_http_server())

    async def stop(self) -> None:
        await asyncio.gather(self._stop_grpc_server(), self._stop_http_server())

    async def _start_grpc_server(self) -> None:
        host, port = self._config.grpc_endpoint.split(":")
        server = grpc.aio.server()
        trace_service_pb2_grpc.add_TraceServiceServicer_to_server(self._trace_service, server)
        logs_service_pb2_grpc.add_LogsServiceServicer_to_server(self._log_service, server)
        metrics_service_pb2_grpc.add_MetricsServiceServicer_to_server(self._metric_service, server)
        try:
            bound_port = server.add_insecure_port(f"{host}:{port}")
        except RuntimeError:
            if port != 0:
                logger.warning(
                    "Failed to bind gRPC server to %s:%s, retrying with ephemeral port.", host, port
                )
                bound_port = server.add_insecure_port(f"{host}:0")
            else:
                raise
        if bound_port == 0:
            raise RuntimeError(f"gRPC server failed to bind to {host}:{port}")
        await server.start()
        self._grpc_wait_task = asyncio.create_task(server.wait_for_termination())
        logger.info("gRPC OTLP server listening on %s:%s", host, bound_port)
        self._config.grpc_endpoint = f"{host}:{bound_port}"
        self._grpc_server = server

    async def _start_http_server(self) -> None:
        host, port_str = self._config.http_endpoint.split(":")
        port = int(port_str)
        app = web.Application()
        app.add_routes(
            [
                web.post("/v1/traces", self._handle_http_traces),
                web.post("/v1/logs", self._handle_http_logs),
                web.post("/v1/metrics", self._handle_http_metrics),
            ]
        )
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, host, port)
            await site.start()
        except OSError:
            if port != 0:
                logger.warning(
                    "Failed to bind HTTP server to %s:%d, retrying with ephemeral port.", host, port
                )
                site = web.TCPSite(runner, host, 0)
                await site.start()
            else:
                raise
        actual_port = site._server.sockets[0].getsockname()[1] if site._server and site._server.sockets else port
        logger.info("HTTP OTLP server listening on %s:%d", host, actual_port)
        self._config.http_endpoint = f"{host}:{actual_port}"
        self._http_runner = runner
        self._http_site = site

    async def _stop_grpc_server(self) -> None:
        if self._grpc_server:
            await self._grpc_server.stop(0)
            self._grpc_server = None
        if self._grpc_wait_task:
            self._grpc_wait_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._grpc_wait_task
            self._grpc_wait_task = None

    async def _stop_http_server(self) -> None:
        if self._http_site:
            await self._http_site.stop()
            self._http_site = None
        if self._http_runner:
            await self._http_runner.cleanup()
            self._http_runner = None

    async def _handle_http_traces(self, request: web.Request) -> web.Response:
        payload = await request.read()
        content_type = request.headers.get("content-type", "")
        message = trace_service_pb2.ExportTraceServiceRequest()
        self._parse_http_payload(payload, content_type, message)
        await self._trace_service.Export(message, None)
        return web.Response(status=200)

    async def _handle_http_logs(self, request: web.Request) -> web.Response:
        payload = await request.read()
        content_type = request.headers.get("content-type", "")
        message = logs_service_pb2.ExportLogsServiceRequest()
        self._parse_http_payload(payload, content_type, message)
        await self._log_service.Export(message, None)
        return web.Response(status=200)

    async def _handle_http_metrics(self, request: web.Request) -> web.Response:
        payload = await request.read()
        content_type = request.headers.get("content-type", "")
        message = metrics_service_pb2.ExportMetricsServiceRequest()
        self._parse_http_payload(payload, content_type, message)
        await self._metric_service.Export(message, None)
        return web.Response(status=200)

    @staticmethod
    def _parse_http_payload(raw: bytes, content_type: str, message) -> None:
        if "json" in content_type:
            json_format.Parse(raw.decode("utf-8"), message)
        else:
            message.ParseFromString(raw)
