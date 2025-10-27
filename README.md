# OTelScope

OTelScope is an interactive, terminal-based OpenTelemetry explorer that lets you follow spans, logs, and metrics as they stream in. Point your application’s OTLP exporter at OTelScope and watch traces assemble in real time with full payload detail only a keystroke away.

## Features

- Drop-in OTLP collector replacement with gRPC and HTTP receivers on familiar ports.
- Live ASCII tree that groups telemetry by trace and preserves span hierarchy.
- Inline access to raw span, log, and metric payloads rendered as readable YAML.
- Keyboard-driven workflow with quick reset for repeat runs and experiments.
- Lightweight Python runtime that can live alongside your local services.

## Quickstart

### Prerequisites

- Python 3.10 or newer.
- An OpenTelemetry-instrumented application capable of exporting OTLP data over gRPC or HTTP.
- `pip install aiohttp grpcio opentelemetry-api opentelemetry-sdk opentelemetry-proto google.protobuf` (install inside a virtual environment for best results).

### Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install project dependencies (see the prerequisite list above).

```bash
git clone https://github.com/your-org/otelscope.git
cd otelscope
python -m venv .venv
source .venv/bin/activate
pip install aiohttp grpcio opentelemetry-api opentelemetry-sdk opentelemetry-proto google.protobuf
```

### Run OTelScope

```bash
python -m otelscope.main --grpc-endpoint 0.0.0.0:4317 --http-endpoint 0.0.0.0:4318
```

Keep the process running, then configure your application’s OTLP exporter to send telemetry to the same endpoints. As data arrives, the terminal UI will update automatically.

### Parse Collector Debug Output

Render previously captured OpenTelemetry Collector debug output as an ASCII tree by pointing to the file:

```bash
python otelscope.py --otelcol_output path/to/collector_output.txt
```

By default all telemetry types (spans, logs, metrics) are shown. Use `--telemetry` to filter the view:

```bash
# Only spans
python otelscope.py --otelcol_output path/to/collector_output.txt --telemetry=span

# Spans and logs (omit metrics)
python otelscope.py --otelcol_output path/to/collector_output.txt --telemetry=span,log

# Logs and metrics only (no spans)
python otelscope.py --otelcol_output path/to/collector_output.txt --telemetry=log,metric
```

Valid telemetry types: `span`, `log`, `metric`. Stdin-based usage and `--only-traces` have been removed.

## Using the UI

- `↑` / `↓` move between nodes.
- `→` or `+` expands the selected item; `←` or `-` collapses it.
- Press `+` or `-` again on spans/logs/metrics to toggle raw payload detail.
- `r` clears accumulated telemetry and resets the view.
- `q` exits gracefully without leaving the terminal in an unusable state.

The header bar summarizes active traces and spans and reminds you which OTLP endpoints are currently exposed.

## Configuration

All runtime configuration is handled via command-line flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--grpc-endpoint` | `127.0.0.1:4317` | Address and port for OTLP gRPC ingestion. |
| `--http-endpoint` | `127.0.0.1:4318` | Address and port for OTLP HTTP ingestion. |
| `--retention` | `100` | Maximum number of traces kept in memory before older entries are pruned. |
| `--log-level` | `INFO` | Logging verbosity for diagnostic output (`DEBUG`, `INFO`, `WARNING`, etc.). |

## Tips & Troubleshooting

- If the UI appears blank, confirm your terminal supports curses and is at least 80×24 characters.
- Run with `--log-level DEBUG` to stream ingestion events and diagnose malformed payloads.
- Use `Ctrl+C` if needed; OTelScope catches the signal and shuts down the receivers and UI cleanly.

### Log Attachment Name Derivation

Earlier versions attempted to read a non-existent `LogRecord.name` field from the OTLP proto and produced an `AttributeError` during ingestion. The server now derives a display name in this order:

1. `event.name` attribute (if present)
2. Common logger name attributes: `logger.name`, `log.logger`, `otel.logger.name`
3. `severity_text`
4. The string `body` value
5. Fallback: `"log"`

This prevents crashes when applications export OTLP logs without naming attributes. If you want richer log node labels in the tree, add an `event.name` attribute via your logging instrumentation.

### Generative AI Operation Labels

When spans or attachments include the attribute `gen_ai.operation.name`, OTelScope appends `[op:<value>]` to the corresponding tree line (e.g., `Span ID: ... - Name: embedding_request [op:embed]`). This makes it easier to visually scan workflows involving multiple generative AI operations. Supported sources:

- Span attributes (e.g., set via custom instrumentation)
- Log / metric attachment attributes (inside `attributes` or `point_attributes` collections)

No special configuration is required—export OTLP telemetry with `gen_ai.operation.name` populated and it will appear automatically.

## Changelog

### Unreleased

- Initial preview with live OTLP ingestion over gRPC/HTTP.
- Real-time terminal UI showing traces, spans, logs, and metrics in a single tree.
- On-demand payload expansion rendered via `google.protobuf.json_format`.
- Configurable trace retention and keyboard-driven reset.
- Graceful handling of OTLP logs without a `name` field; robust fallback naming strategy to avoid `AttributeError`.
- Display of `gen_ai.operation.name` for spans and attachments when present.

## Feature Roadmap

- **Packaging & Distribution** – Provide an installable Python package and CLI entry point for easy distribution.
- **Ingestion Enhancements** – Harden the gRPC/HTTP receivers, broaden OTLP compatibility, and support additional content types.
- **Trace Store Improvements** – Improve merge logic for out-of-order data, pruning strategies, and attachment handling.
- **UI Enhancements** – Add richer navigation aids, filtering, and accessory views while maintaining the terminal focus.
- **Runtime Coordination** – Offer presets for embedding OTelScope alongside other asyncio workloads or containerized stacks.
- **Testing & Tooling** – Expand unit and integration test coverage with reusable telemetry fixtures.
- **Stretch Goals** – Introduce search and filtering, optional persistence for replay, and trace/service-level filters.
