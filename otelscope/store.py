"""
State management for OTelScope.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from .models import Attachment, SpanNode, TraceRecord


class TraceStore:
    """In-memory storage for trace-centric telemetry."""

    def __init__(
        self,
        *,
        retention_traces: int = 100,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._traces: Dict[str, TraceRecord] = {}
        self._order: Deque[str] = deque()
        self._retention_traces = retention_traces
        self._lock = asyncio.Lock()
        self._listeners: List[asyncio.Queue] = []
        self._loop = loop or asyncio.get_event_loop()

    async def add_span(self, span: SpanNode) -> None:
        async with self._lock:
            trace = self._get_or_create_trace(span.trace_id)
            existing = trace.spans.get(span.span_id)
            if existing:
                # Merge attachments and children references if we receive span updates.
                span.attachments.extend(existing.attachments)
                span.children.extend(existing.children)
            trace.add_span(span)
            self._ensure_hierarchy(trace)
            self._record_order(span.trace_id)
        await self._notify_listeners(span.trace_id)

    async def add_attachment(self, attachment: Attachment) -> None:
        trace_id = attachment.trace_id
        if not trace_id:
            return
        async with self._lock:
            trace = self._get_or_create_trace(trace_id)
            target_span = trace.spans.get(attachment.span_id) if attachment.span_id else None
            if target_span:
                target_span.add_attachment(attachment)
            else:
                trace.add_orphan_attachment(attachment)
            self._record_order(trace_id)
        await self._notify_listeners(trace_id)

    async def reset(self) -> None:
        async with self._lock:
            self._traces.clear()
            self._order.clear()
        await self._notify_listeners(None)

    async def snapshot(self) -> Dict[str, TraceRecord]:
        async with self._lock:
            # Return shallow copies to avoid mutation outside the lock.
            return {trace_id: self._clone_trace(trace) for trace_id, trace in self._traces.items()}

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._listeners.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._listeners:
            self._listeners.remove(queue)

    def _get_or_create_trace(self, trace_id: str) -> TraceRecord:
        trace = self._traces.get(trace_id)
        if not trace:
            trace = TraceRecord(trace_id=trace_id)
            self._traces[trace_id] = trace
        return trace

    def _record_order(self, trace_id: str) -> None:
        if trace_id in self._order:
            self._order.remove(trace_id)
        self._order.append(trace_id)
        while len(self._order) > self._retention_traces:
            oldest = self._order.popleft()
            self._traces.pop(oldest, None)

    async def _notify_listeners(self, trace_id: Optional[str]) -> None:
        stale: List[asyncio.Queue] = []
        for queue in list(self._listeners):
            try:
                queue.put_nowait(trace_id)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            self.unsubscribe(queue)

    @staticmethod
    def _clone_trace(trace: TraceRecord) -> TraceRecord:
        cloned = TraceRecord(
            trace_id=trace.trace_id,
            spans={},
            orphan_attachments=list(trace.orphan_attachments),
            received_at=trace.received_at,
        )
        for span_id, span in trace.spans.items():
            cloned_span = replace(span, attachments=list(span.attachments), children=list(span.children))
            cloned.spans[span_id] = cloned_span
        return cloned

    @staticmethod
    def _ensure_hierarchy(trace: TraceRecord) -> None:
        # Reset children lists before rebuilding relationships.
        for span in trace.spans.values():
            span.children.clear()

        for span in trace.spans.values():
            parent_id = span.parent_span_id
            if parent_id and parent_id in trace.spans:
                trace.spans[parent_id].children.append(span)
