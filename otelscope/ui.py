"""
Interactive curses-based UI for the OTelScope telemetry explorer.
"""

from __future__ import annotations

import asyncio
import curses
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from google.protobuf import json_format

from .models import Attachment, SpanNode, TraceRecord
from .server import TelemetryServerConfig
from .store import TraceStore

logger = logging.getLogger(__name__)


TreeKey = Tuple[str, ...]


@dataclass
class TreeEntry:
    """Represents a renderable node in the tree."""

    key: TreeKey
    label: str
    prefix: str
    level: int
    node_type: str
    has_children: bool
    has_detail: bool
    payload: Any
    summary: Optional[str]
    parent_key: Optional[TreeKey]

    @property
    def display_text(self) -> str:
        return f"{self.prefix}{self.label}"


class OTelScopeUI:
    """Curses-based UI for navigating telemetry traces."""

    def __init__(
        self,
        store: TraceStore,
        config: TelemetryServerConfig,
        *,
        on_exit: Optional[Callable[[], None]] = None,
    ) -> None:
        self._store = store
        self._config = config
        self._on_exit = on_exit

        self._stop_event = asyncio.Event()
        self._subscription: Optional[asyncio.Queue] = None

        self._stdscr: Optional["curses._CursesWindow"] = None
        self._needs_render = True
        self._entries: List[TreeEntry] = []
        self._expanded: Set[TreeKey] = set()
        self._detail_expanded: Set[TreeKey] = set()
        self._detail_cache: Dict[TreeKey, List[str]] = {}
        self._known_keys: Set[TreeKey] = set()

        self._selected_index = 0
        self._scroll_offset = 0
        self._counts: Dict[str, int] = {"traces": 0, "spans": 0, "logs": 0, "metrics": 0}

    async def run(self) -> None:
        if self._subscription is not None:
            logger.warning("OTelScope UI already running")
            return

        self._subscription = self._store.subscribe()
        self._init_curses()

        try:
            while not self._stop_event.is_set():
                updated = await self._drain_updates()
                if updated:
                    await self._refresh_snapshot()
                if self._needs_render:
                    self._render()
                await self._process_input()
                await asyncio.sleep(0.05)
        finally:
            if self._subscription is not None:
                self._store.unsubscribe(self._subscription)
                self._subscription = None
            self._teardown_curses()

    async def stop(self) -> None:
        self._stop_event.set()

    async def _drain_updates(self) -> bool:
        if not self._subscription:
            return False

        updated = False
        while True:
            try:
                self._subscription.get_nowait()
                updated = True
            except asyncio.QueueEmpty:
                break
        return updated

    async def _refresh_snapshot(self) -> None:
        snapshot = await self._store.snapshot()
        self._counts = self._compute_counts(snapshot)
        self._entries = self._build_entries(snapshot)

        current_keys = {entry.key for entry in self._entries}
        self._expanded = {key for key in self._expanded if key in current_keys}
        self._detail_expanded = {key for key in self._detail_expanded if key in current_keys}
        self._detail_cache = {key: value for key, value in self._detail_cache.items() if key in current_keys}
        self._known_keys = current_keys

        if self._selected_index >= len(self._entries):
            self._selected_index = max(len(self._entries) - 1, 0)
        self._needs_render = True

    async def _process_input(self) -> None:
        if not self._stdscr:
            return

        try:
            ch = self._stdscr.getch()
        except curses.error:
            return

        if ch == -1:
            return

        if ch in (ord("q"), ord("Q")):
            await self._handle_quit()
            return

        if ch in (ord("r"), ord("R")):
            await self._handle_reset()
            return

        if ch in (curses.KEY_UP, ord("k")):
            self._move_selection(-1)
        elif ch in (curses.KEY_DOWN, ord("j")):
            self._move_selection(1)
        elif ch in (curses.KEY_RIGHT, ord("+"), ord("=")):
            await self._expand_selected()
        elif ch in (curses.KEY_LEFT, ord("-"), ord("_")):
            await self._collapse_selected()
        elif ch == curses.KEY_RESIZE:
            self._needs_render = True

    async def _handle_quit(self) -> None:
        if self._on_exit:
            self._on_exit()
        self._stop_event.set()

    async def _handle_reset(self) -> None:
        await self._store.reset()
        self._expanded.clear()
        self._detail_expanded.clear()
        self._detail_cache.clear()
        self._selected_index = 0
        self._scroll_offset = 0

    def _move_selection(self, delta: int) -> None:
        if not self._entries:
            return
        self._selected_index = max(0, min(self._selected_index + delta, len(self._entries) - 1))
        max_y, _ = self._stdscr.getmaxyx() if self._stdscr else (0, 0)
        viewport_height = max(max_y - 3, 1)
        if self._selected_index < self._scroll_offset:
            self._scroll_offset = self._selected_index
        elif self._selected_index >= self._scroll_offset + viewport_height:
            self._scroll_offset = self._selected_index - viewport_height + 1
        self._needs_render = True

    async def _expand_selected(self) -> None:
        entry = self._current_entry()
        if not entry:
            return
        key = entry.key

        if entry.has_children and key not in self._expanded:
            self._expanded.add(key)
            self._needs_render = True
            return

        if entry.has_detail:
            if key not in self._detail_expanded:
                self._detail_expanded.add(key)
                self._needs_render = True

    async def _collapse_selected(self) -> None:
        entry = self._current_entry()
        if not entry:
            return
        key = entry.key

        if key in self._detail_expanded:
            self._detail_expanded.remove(key)
            self._needs_render = True
            return

        if key in self._expanded:
            self._expanded.remove(key)
            self._needs_render = True
            return

        if entry.parent_key is not None:
            parent_index = self._find_entry_index(entry.parent_key)
            if parent_index is not None:
                self._selected_index = parent_index
                self._move_selection(0)

    def _current_entry(self) -> Optional[TreeEntry]:
        if not self._entries:
            return None
        return self._entries[self._selected_index]

    def _find_entry_index(self, key: TreeKey) -> Optional[int]:
        for idx, entry in enumerate(self._entries):
            if entry.key == key:
                return idx
        return None

    def _build_entries(self, snapshot: Dict[str, TraceRecord]) -> List[TreeEntry]:
        entries: List[TreeEntry] = []
        for trace_id in sorted(snapshot.keys()):
            trace = snapshot[trace_id]
            trace_key: TreeKey = ("trace", trace_id)
            label = f"Trace ID: {trace_id}"
            has_children = bool(trace.spans or trace.orphan_attachments)
            entry = TreeEntry(
                key=trace_key,
                label=label,
                prefix="",
                level=0,
                node_type="trace",
                has_children=has_children,
                has_detail=False,
                payload=None,
                summary=None,
                parent_key=None,
            )
            entries.append(entry)
            if trace_key not in self._known_keys and has_children:
                self._expanded.add(trace_key)
            if self._is_expanded(trace_key):
                children = self._build_trace_children(trace, parent_key=trace_key, parent_stack=[])
                entries.extend(children)
        return entries

    def _build_trace_children(
        self,
        trace: TraceRecord,
        *,
        parent_key: TreeKey,
        parent_stack: List[bool],
    ) -> List[TreeEntry]:
        entries: List[TreeEntry] = []
        roots = self._find_root_spans(trace)

        # If multiple disjoint roots exist, synthesize a workflow/agent root for display
        children_items: List[Tuple[str, Any]] = []
        if len(roots) > 1:
            workflow_id = None
            agent_name = None
            # Scan span attributes for workflow / agent hints
            for span in trace.spans.values():
                if not workflow_id:
                    workflow_id = span.attributes.get("gen_ai.workflow.id")
                if not agent_name:
                    agent_name = span.attributes.get("agent_name") or span.attributes.get("gen_ai.agent.name")
                if workflow_id and agent_name:
                    break
            if not workflow_id or not agent_name:
                # Scan attachments (metric/log payloads) for workflow / agent attributes
                for span in trace.spans.values():
                    for att in span.attachments:
                        payload = att.raw_payload
                        if isinstance(payload, dict):
                            att_attrs = payload.get("attributes") or payload.get("point_attributes") or {}
                            if not workflow_id:
                                workflow_id = att_attrs.get("gen_ai.workflow.id")
                            if not agent_name:
                                agent_name = att_attrs.get("agent_name") or att_attrs.get("gen_ai.agent.name")
                            if workflow_id and agent_name:
                                break
                    if workflow_id and agent_name:
                        break
            synthetic_label_parts = []
            if workflow_id:
                synthetic_label_parts.append(f"workflow {workflow_id}")
            if agent_name:
                synthetic_label_parts.append(f"agent {agent_name}")
            synthetic_name = " / ".join(synthetic_label_parts) if synthetic_label_parts else "workflow"
            children_items.append(("synthetic-root", {"name": synthetic_name, "roots": roots}))
        else:
            children_items.extend(("span", span) for span in roots)
        if trace.orphan_attachments:
            children_items.append(("orphan", trace.orphan_attachments))

        for index, (item_type, item) in enumerate(children_items):
            is_last = index == len(children_items) - 1
            stack = parent_stack + [is_last]
            if item_type == "span":
                entries.extend(self._build_span_branch(trace.trace_id, item, parent_key, stack))
            elif item_type == "synthetic-root":
                entries.extend(self._build_synthetic_root_branch(trace.trace_id, item, parent_key, stack))
            else:
                entries.extend(self._build_orphan_branch(trace.trace_id, item, parent_key, stack))
        return entries

    def _build_span_branch(
        self,
        trace_id: str,
        span: SpanNode,
        parent_key: TreeKey,
        stack: List[bool],
    ) -> List[TreeEntry]:
        entries: List[TreeEntry] = []
        span_key: TreeKey = ("span", trace_id, span.span_id)
        prefix = self._prefix_from_stack(stack)
        parent_display = span.parent_span_id or "none"
        label = f"Span ID: {span.span_id} (Parent: {parent_display}) - Name: {span.name}"
        op_name = span.attributes.get("gen_ai.operation.name")
        if op_name:
            label = f"{label} [op:{op_name}]"
        has_children = bool(span.children or span.attachments)
        entry = TreeEntry(
            key=span_key,
            label=label,
            prefix=prefix,
            level=len(stack),
            node_type="span",
            has_children=has_children,
            has_detail=True,
            payload=span.raw_payload,
            summary=None,
            parent_key=parent_key,
        )
        entries.append(entry)
        if span_key not in self._known_keys and has_children:
            self._expanded.add(span_key)
        if self._is_expanded(span_key):
            attachments = sorted(span.attachments, key=lambda a: (a.attachment_type, a.name))
            child_items: List[Tuple[str, Any]] = [("attachment", attachment) for attachment in attachments]
            child_spans = sorted(span.children, key=lambda child: child.start_time_unix_nano)
            child_items += [("span", child) for child in child_spans]
            for index, (kind, child) in enumerate(child_items):
                is_last = index == len(child_items) - 1
                child_stack = stack + [is_last]
                if kind == "span":
                    entries.extend(self._build_span_branch(trace_id, child, span_key, child_stack))
                else:
                    entries.append(
                        self._build_attachment_entry(trace_id, child, span_key, child_stack)
                    )
        return entries

    def _build_attachment_entry(
        self,
        trace_id: str,
        attachment: Attachment,
        parent_key: TreeKey,
        stack: List[bool],
    ) -> TreeEntry:
        key: TreeKey = ("attachment", trace_id, attachment.attachment_id)
        label = f"{attachment.attachment_type.capitalize()}: {attachment.name}"
        raw = attachment.raw_payload
        if isinstance(raw, dict):
            attrs = raw.get("attributes") or raw.get("point_attributes") or {}
            op_name = attrs.get("gen_ai.operation.name")
            if op_name:
                label = f"{label} [op:{op_name}]"
        prefix = self._prefix_from_stack(stack)
        return TreeEntry(
            key=key,
            label=label,
            prefix=prefix,
            level=len(stack),
            node_type=attachment.attachment_type,
            has_children=False,
            has_detail=True,
            payload=attachment.raw_payload,
            summary=attachment.summary,
            parent_key=parent_key,
        )

    def _build_orphan_branch(
        self,
        trace_id: str,
        attachments: Sequence[Attachment],
        parent_key: TreeKey,
        stack: List[bool],
    ) -> List[TreeEntry]:
        entries: List[TreeEntry] = []
        group_key: TreeKey = ("attachment-group", trace_id, "orphan")
        prefix = self._prefix_from_stack(stack)
        label = "Attachments without span"
        entry = TreeEntry(
            key=group_key,
            label=label,
            prefix=prefix,
            level=len(stack),
            node_type="attachment-group",
            has_children=bool(attachments),
            has_detail=False,
            payload=None,
            summary=None,
            parent_key=parent_key,
        )
        entries.append(entry)
        if group_key not in self._known_keys and attachments:
            self._expanded.add(group_key)
        if self._is_expanded(group_key):
            sorted_attachments = sorted(attachments, key=lambda a: (a.attachment_type, a.name))
            for index, attachment in enumerate(sorted_attachments):
                is_last = index == len(sorted_attachments) - 1
                child_stack = stack + [is_last]
                entries.append(self._build_attachment_entry(trace_id, attachment, group_key, child_stack))
        return entries

    def _build_synthetic_root_branch(
        self,
        trace_id: str,
        data: Dict[str, Any],
        parent_key: TreeKey,
        stack: List[bool],
    ) -> List[TreeEntry]:
        """Render a synthetic workflow/agent root grouping disjoint spans."""
        entries: List[TreeEntry] = []
        key: TreeKey = ("synthetic-root", trace_id)
        prefix = self._prefix_from_stack(stack)
        label = f"Root: {data['name']}"
        roots: List[SpanNode] = data["roots"]
        entry = TreeEntry(
            key=key,
            label=label,
            prefix=prefix,
            level=len(stack),
            node_type="synthetic-root",
            has_children=bool(roots),
            has_detail=False,
            payload=None,
            summary=None,
            parent_key=parent_key,
        )
        entries.append(entry)
        if key not in self._known_keys and roots:
            self._expanded.add(key)
        if self._is_expanded(key):
            for idx, span in enumerate(sorted(roots, key=lambda s: s.start_time_unix_nano)):
                is_last = idx == len(roots) - 1
                child_stack = stack + [is_last]
                entries.extend(self._build_span_branch(trace_id, span, key, child_stack))
        return entries

    def _find_root_spans(self, trace: TraceRecord) -> List[SpanNode]:
        roots = [
            span
            for span in trace.spans.values()
            if not span.parent_span_id or span.parent_span_id not in trace.spans
        ]
        return sorted(roots, key=lambda span: span.start_time_unix_nano)

    def _is_expanded(self, key: TreeKey) -> bool:
        return key in self._expanded

    def _prefix_from_stack(self, stack: List[bool]) -> str:
        if not stack:
            return ""
        prefix_parts: List[str] = []
        for is_last in stack[:-1]:
            prefix_parts.append("    " if is_last else "│   ")
        prefix_parts.append("└── " if stack[-1] else "├── ")
        return "".join(prefix_parts)

    def _render(self) -> None:
        if not self._stdscr:
            return

        self._stdscr.erase()
        max_y, max_x = self._stdscr.getmaxyx()

        header = (
            f"OTelScope | Traces: {self._counts['traces']} "
            f"Spans: {self._counts['spans']} Logs: {self._counts['logs']} Metrics: {self._counts['metrics']} "
            f"| gRPC {self._config.grpc_endpoint} | HTTP {self._config.http_endpoint}"
        )
        footer = "Controls: ↑/↓ move · ← collapse · → expand · +/- details · r reset · q quit"

        try:
            self._stdscr.addnstr(0, 0, header, max_x - 1, curses.A_BOLD)
        except curses.error:
            pass

        content_height = max_y - 2
        self._render_entries(content_height, max_x)

        try:
            self._stdscr.addnstr(max_y - 1, 0, footer, max_x - 1, curses.A_DIM)
        except curses.error:
            pass

        self._stdscr.refresh()
        self._needs_render = False

    def _render_entries(self, max_rows: int, max_x: int) -> None:
        if not self._stdscr:
            return

        row = 1
        start = self._scroll_offset
        end = min(start + max_rows - 1, len(self._entries))

        for index in range(start, end):
            entry = self._entries[index]
            attr = curses.A_REVERSE if index == self._selected_index else curses.A_NORMAL
            line = entry.display_text
            if entry.summary and index != self._selected_index:
                line = f"{line}  [{entry.summary}]"
            try:
                self._stdscr.addnstr(row, 0, line, max_x - 1, attr)
            except curses.error:
                pass
            row += 1

            if entry.key in self._detail_expanded:
                detail_lines = self._detail_cache.get(entry.key)
                if detail_lines is None:
                    detail_lines = self._build_detail_lines(entry, max_x - 4)
                    self._detail_cache[entry.key] = detail_lines
                indent = "    " * (entry.level + 1)
                for detail_line in detail_lines:
                    if row >= max_rows:
                        break
                    try:
                        self._stdscr.addnstr(row, 0, f"{indent}{detail_line}", max_x - 1, curses.A_DIM)
                    except curses.error:
                        pass
                    row += 1

        # Fill remaining rows with empty space
        while row < max_rows:
            try:
                self._stdscr.addnstr(row, 0, "", max_x - 1)
            except curses.error:
                pass
            row += 1

    def _build_detail_lines(self, entry: TreeEntry, max_width: int) -> List[str]:
        payload = entry.payload
        if payload is None:
            return []

        if hasattr(payload, "DESCRIPTOR"):
            data = json_format.MessageToDict(payload, preserving_proto_field_name=True)
        else:
            data = payload

        yaml_lines = _to_yaml_lines(data)
        wrapped: List[str] = []
        for raw_line in yaml_lines:
            if len(raw_line) <= max_width:
                wrapped.append(raw_line)
            else:
                wrapped.extend([raw_line[i : i + max_width] for i in range(0, len(raw_line), max_width)])
        return wrapped

    def _init_curses(self) -> None:
        self._stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self._stdscr.nodelay(True)
        self._stdscr.keypad(True)

    def _teardown_curses(self) -> None:
        if not self._stdscr:
            return
        try:
            self._stdscr.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
        finally:
            self._stdscr = None

    def _compute_counts(self, snapshot: Dict[str, TraceRecord]) -> Dict[str, int]:
        spans = 0
        logs = 0
        metrics = 0
        for trace in snapshot.values():
            spans += len(trace.spans)
            for span in trace.spans.values():
                logs += sum(1 for att in span.attachments if att.attachment_type == "log")
                metrics += sum(1 for att in span.attachments if att.attachment_type == "metric")
            logs += sum(1 for att in trace.orphan_attachments if att.attachment_type == "log")
            metrics += sum(1 for att in trace.orphan_attachments if att.attachment_type == "metric")
        return {
            "traces": len(snapshot),
            "spans": spans,
            "logs": logs,
            "metrics": metrics,
        }


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None)))


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    text = str(value)
    if not text or any(ch in text for ch in [":", "#", "-", "\n", "'"]) or text != text.strip():
        text = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{text}"'
    return text


def _format_key(key: Any) -> str:
    if isinstance(key, str) and key and all(ch.isalnum() or ch in ("_", "-") for ch in key):
        return key
    return _format_scalar(key)


def _to_yaml_lines(value: Any, indent: int = 0) -> List[str]:
    prefix = "  " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: List[str] = []
        for key, item in value.items():
            formatted_key = _format_key(key)
            if _is_scalar(item) or isinstance(item, bytes):
                lines.append(f"{prefix}{formatted_key}: {_format_scalar(item)}")
            else:
                lines.append(f"{prefix}{formatted_key}:")
                lines.extend(_to_yaml_lines(item, indent + 1))
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines: List[str] = []
        for item in value:
            if _is_scalar(item) or isinstance(item, bytes):
                lines.append(f"{prefix}- {_format_scalar(item)}")
            else:
                lines.append(f"{prefix}-")
                lines.extend(_to_yaml_lines(item, indent + 1))
        return lines

    return [f"{prefix}{_format_scalar(value)}"]
