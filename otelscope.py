import argparse
import re
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Iterable, Tuple

# Define data structures
Span = namedtuple("Span", ["trace_id", "span_id", "parent_id", "name", "type"])
Log = namedtuple("Log", ["trace_id", "span_id", "name", "type"])
Metric = namedtuple("Metric", ["trace_id", "span_id", "name", "type"])

_CONTROL_CHARS = "".join(chr(i) for i in range(32) if i not in (9, 10, 13))
_CONTROL_TRANS = str.maketrans("", "", _CONTROL_CHARS)


def sanitize_line(line):
    """Remove ANSI control characters while preserving indentation."""
    return line.translate(_CONTROL_TRANS)


def detect_section(line):
    """Return the current top-level signal section (Logs/Traces/Metrics) if present."""
    parts = line.split("\t")
    if len(parts) >= 3 and parts[1] == "info":
        return parts[2]
    return None


def _parse_log_record(lines, start_idx, logs):
    event_name = None
    trace_id = None
    span_id = None
    op_name = None
    i = start_idx + 1

    while i < len(lines):
        stripped = lines[i].strip()
        section = detect_section(lines[i])
        if stripped.startswith("LogRecord #") or section in {"Logs", "Traces", "Metrics"}:
            break

        if "EventName:" in stripped:
            event_name = stripped.split("EventName:", 1)[1].strip()
        elif "gen_ai.operation.name:" in stripped:
            # Attribute line format: -> gen_ai.operation.name: Str(chat)
            match = re.search(r"gen_ai\.operation\.name:\s*Str\(([^)]+)\)", stripped)
            if match:
                op_name = match.group(1)
        elif "Trace ID" in stripped:
            match = re.search(r"Trace ID\s*:\s*([0-9a-fA-F]+)", stripped)
            if match:
                trace_id = match.group(1)
        elif "Span ID" in stripped:
            match = re.search(r"Span ID\s*:\s*([0-9a-fA-F]+)", stripped)
            if match:
                span_id = match.group(1)

        i += 1

    if event_name and trace_id and span_id:
        name = f"{event_name} [op:{op_name}]" if op_name else event_name
        logs.append(Log(trace_id, span_id, name, "log"))

    return i


def parse_logs(lines):
    logs = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("LogRecord #"):
            i = _parse_log_record(lines, i, logs)
            continue
        i += 1
    return logs


def _parse_span_block(lines, start_idx, spans):
    trace_id = None
    parent_id = None
    span_id = None
    name = None
    op_name = None
    i = start_idx + 1

    while i < len(lines):
        stripped = lines[i].strip()
        section = detect_section(lines[i])
        if stripped.startswith("Span #") or section in {"Logs", "Traces", "Metrics"}:
            break

        if stripped.startswith("Trace ID"):
            match = re.search(r"Trace ID\s*:\s*([0-9a-fA-F]+)", stripped)
            if match:
                trace_id = match.group(1)
        elif stripped.startswith("Parent ID"):
            match = re.search(r"Parent ID\s*:\s*([0-9a-fA-F]*)", stripped)
            if match:
                parent_id = match.group(1) or None
        elif re.match(r"^ID\s*:", stripped):
            match = re.search(r"ID\s*:\s*([0-9a-fA-F]+)", stripped)
            if match:
                span_id = match.group(1)
        elif stripped.startswith("Name"):
            match = re.search(r"Name\s*:\s*(.+)", stripped)
            if match:
                name = match.group(1).strip()
        elif "gen_ai.operation.name:" in stripped:
            match = re.search(r"gen_ai\.operation\.name:\s*Str\(([^)]+)\)", stripped)
            if match:
                op_name = match.group(1)

        i += 1

    if trace_id and span_id and name:
        final_name = f"{name} [op:{op_name}]" if op_name else name
        spans.append(Span(trace_id, span_id, parent_id, final_name, "span"))

    return i


def parse_spans(lines):
    spans = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("Span #"):
            i = _parse_span_block(lines, i, spans)
            continue
        i += 1
    return spans


def parse_metrics(lines):
    metrics = []
    metric_name = None
    token_type = None
    op_name = None
    in_metric = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        section = detect_section(line)

        if stripped.startswith("Metric #"):
            in_metric = True
            metric_name = None
            token_type = None
            op_name = None
            continue

        if section and section != "Metrics":
            in_metric = False
            metric_name = None
            token_type = None
            continue

        if not in_metric:
            continue

        if "-> Name:" in stripped and metric_name is None:
            metric_name = stripped.split("-> Name:", 1)[1].strip()
            continue

        if stripped == "Data point attributes:":
            token_type = None
            op_name = None
            continue

        if "gen_ai.token.type:" in stripped:
            match = re.search(r"gen_ai\.token\.type:\s*Str\(([^)]+)\)", stripped)
            if match:
                token_type = match.group(1)
            continue
        if "gen_ai.operation.name:" in stripped:
            match = re.search(r"gen_ai\.operation\.name:\s*Str\(([^)]+)\)", stripped)
            if match:
                op_name = match.group(1)
            continue

        if stripped.startswith("Exemplar #") and metric_name:
            trace_id = None
            span_id = None
            j = idx + 1

            while j < len(lines):
                exemplar_line = lines[j].strip()
                if exemplar_line.startswith("-> Trace ID:"):
                    match = re.search(r"Trace ID:\s*([0-9a-fA-F]+)", exemplar_line)
                    if match:
                        trace_id = match.group(1)
                elif exemplar_line.startswith("-> Span ID:"):
                    match = re.search(r"Span ID:\s*([0-9a-fA-F]+)", exemplar_line)
                    if match:
                        span_id = match.group(1)
                elif not exemplar_line.startswith("->"):
                    break
                j += 1

            if trace_id and span_id:
                parts = []
                if token_type:
                    parts.append(f"({token_type})")
                if op_name:
                    parts.append(f"[op:{op_name}]")
                suffix = " " + " ".join(parts) if parts else ""
                metrics.append(Metric(trace_id, span_id, f"{metric_name}{suffix}", "metric"))

    return metrics


def parse_telemetry_output(raw_text: str) -> Tuple[list[Span], list[Log], list[Metric]]:
    """Parse OpenTelemetry collector debug output from provided text."""
    if not raw_text.strip():
        return [], [], []

    lines = [sanitize_line(line) for line in raw_text.splitlines()]
    spans = parse_spans(lines)
    logs = parse_logs(lines)
    metrics = parse_metrics(lines)

    return spans, logs, metrics


def build_trace_trees(spans, logs, metrics):
    """Build tree structures for each trace ID based on spans' parent-child relationships."""
    trace_data = defaultdict(lambda: {"spans": [], "attachments": defaultdict(list)})

    for span in spans:
        trace_data[span.trace_id]["spans"].append(span)
    for log in logs:
        trace_data[log.trace_id]["attachments"][log.span_id].append(log)
    for metric in metrics:
        trace_data[metric.trace_id]["attachments"][metric.span_id].append(metric)

    trees = {}
    for trace_id, data in trace_data.items():
        span_nodes = {}
        attachments = data["attachments"]

        for span in data["spans"]:
            span_nodes[span.span_id] = {
                "span": span,
                "children": [],
                "attachments": [],
            }

        # Attach metrics/logs to spans where possible
        for span_id, node in span_nodes.items():
            node["attachments"] = attachments.pop(span_id, [])

        # Build the hierarchy
        roots = []
        for span in data["spans"]:
            node = span_nodes[span.span_id]
            parent_id = span.parent_id
            if parent_id and parent_id in span_nodes:
                span_nodes[parent_id]["children"].append(node)
            else:
                roots.append(node)

        # Create pseudo nodes for attachments with missing spans
        for span_id, orphan_attachments in attachments.items():
            roots.append(
                {
                    "span": None,
                    "span_id": span_id or "unknown",
                    "children": [],
                    "attachments": orphan_attachments,
                }
            )

        trees[trace_id] = roots

    return trees


def render_ascii_tree(trees):
    """Render the trees in ASCII format for each trace ID."""
    lines = []
    type_order = {"log": 0, "metric": 1}

    def sort_attachments(items):
        return sorted(items, key=lambda item: (type_order.get(item.type, 99), item.name))

    def render_node(node, prefix, is_last):
        connector = "└──" if is_last else "├──"
        span = node.get("span")

        if span:
            parent_display = span.parent_id or "none"
            lines.append(
                f"{prefix}{connector} Span ID: {span.span_id} (Parent: {parent_display}) - Name: {span.name} (Type: {span.type})"
            )
            child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
            children = []
            for attachment in sort_attachments(node.get("attachments", [])):
                children.append(("attachment", attachment))
            for child in node.get("children", []):
                children.append(("child", child))

            for index, (kind, payload) in enumerate(children):
                last_child = index == len(children) - 1
                if kind == "attachment":
                    label = f"{payload.type.capitalize()}: {payload.name} (Type: {payload.type})"
                    lines.append(
                        f"{child_prefix}{'└──' if last_child else '├──'} {label}"
                    )
                else:
                    render_node(payload, child_prefix, last_child)
        else:
            span_id = node.get("span_id")
            title = (
                f"Attachments for span {span_id}"
                if span_id not in (None, "", "unknown")
                else "Attachments without span"
            )
            lines.append(f"{prefix}{connector} {title}")
            child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
            attachments = sort_attachments(node.get("attachments", []))
            for index, attachment in enumerate(attachments):
                last_child = index == len(attachments) - 1
                label = f"{attachment.type.capitalize()}: {attachment.name} (Type: {attachment.type})"
                lines.append(
                    f"{child_prefix}{'└──' if last_child else '├──'} {label}"
                )

    for trace_id in sorted(trees.keys()):
        lines.append(f"Trace ID: {trace_id}")
        forest = trees[trace_id]
        if not forest:
            lines.append("└── (no spans)")
        else:
            for index, node in enumerate(forest):
                render_node(node, "", index == len(forest) - 1)
        lines.append("")

    return "\n".join(lines).rstrip()


def _load_raw_text(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """Read telemetry text from the provided file path."""
    file_path = args.otelcol_output
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        parser.error(f"Input file not found: {file_path}")
    except OSError as exc:
        parser.error(f"Unable to read '{file_path}': {exc}")
    return ""  # Unreachable due to parser.error


def _parse_cli_args(argv: Iterable[str] | None = None) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Render OpenTelemetry Collector debug output as an ASCII trace tree."
    )
    parser.add_argument(
        "--otelcol_output",
        required=True,
        help="Path to a file containing OpenTelemetry Collector debug output (required).",
    )
    parser.add_argument(
        "--telemetry",
        default="span,log,metric",
        help="Comma-separated list of telemetry types to include: span,log,metric (default: all)",
    )
    return parser, parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None):
    parser, args = _parse_cli_args(argv)
    raw_text = _load_raw_text(args, parser)

    spans, logs, metrics = parse_telemetry_output(raw_text)

    # Parse telemetry filter
    requested = {t.strip().lower() for t in args.telemetry.split(',') if t.strip()}
    valid = {"span", "log", "metric"}
    if not requested:
        requested = valid
    invalid = requested - valid
    if invalid:
        parser.error(f"Invalid telemetry type(s): {', '.join(sorted(invalid))}. Valid types: span, log, metric")

    if "span" not in requested:
        spans = []
    if "log" not in requested:
        logs = []
    if "metric" not in requested:
        metrics = []

    if not (spans or logs or metrics):
        print("No telemetry parsed from input with the provided filter.")
        return

    trees = build_trace_trees(spans, logs, metrics)
    ascii_rep = render_ascii_tree(trees)
    print(ascii_rep)


if __name__ == "__main__":
    main()
