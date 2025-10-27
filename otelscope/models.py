"""
Data models used by OTelScope for representing spans, logs, and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class Attachment:
    """Generic attachment associated with a span."""

    attachment_id: str
    span_id: Optional[str]
    trace_id: Optional[str]
    attachment_type: str  # "log" or "metric"
    name: str
    summary: str
    raw_payload: Any
    received_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class SpanNode:
    """Representation of a span with metadata required for the tree view."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str
    start_time_unix_nano: int
    end_time_unix_nano: int
    attributes: Dict[str, Any]
    status_message: Optional[str]
    status_code: Optional[str]
    raw_payload: Any
    attachments: List[Attachment] = field(default_factory=list)
    children: List["SpanNode"] = field(default_factory=list)
    received_at: datetime = field(default_factory=datetime.utcnow)

    def add_attachment(self, attachment: Attachment) -> None:
        self.attachments.append(attachment)

    def add_child(self, child: "SpanNode") -> None:
        self.children.append(child)


@dataclass(slots=True)
class TraceRecord:
    """Top-level container for all telemetry associated with a trace."""

    trace_id: str
    spans: Dict[str, SpanNode] = field(default_factory=dict)
    orphan_attachments: List[Attachment] = field(default_factory=list)
    received_at: datetime = field(default_factory=datetime.utcnow)

    def add_span(self, span: SpanNode) -> None:
        self.spans[span.span_id] = span
        self.received_at = datetime.utcnow()

    def add_orphan_attachment(self, attachment: Attachment) -> None:
        self.orphan_attachments.append(attachment)
        self.received_at = datetime.utcnow()
