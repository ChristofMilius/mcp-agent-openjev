# mcp-agent-openjev

Local **typed probabilistic decision service** — `Choice`, `Noul`, `Score` with
calibrated probabilities and abstention — powered by
[OpenJev](https://github.com/zhangcy122/OpenJev) over an OpenAI-compatible chat
endpoint, primarily your local **LM Studio**.

OpenJev's stock clients target Ollama (`/api/chat`) and the legacy
`/v1/completions` endpoint. LM Studio serves neither — it exposes
`/v1/chat/completions` with `logprobs`. This project is the adapter: it keeps
OpenJev's schemas, temperature calibration and abstention semantics, but scores
candidate letters through the logprob path LM Studio actually exports. It also
adds the `score` decision OpenJev's client omits.

For whom: anyone with a local OpenAI-compatible model server who wants
System-1 style typed decisions (routing, guardrails, intent detection,
severity ratings) without a Jev subscription — and agents that want them as an
MCP tool.

## How it works

For a `Choice` over N options, the client:

1. labels the options `A`, `B`, `C`, … and asks the model to reply with one letter, `max_tokens=1`;
2. reads the letter logits from `choices[0].logprobs.content[0].top_logprobs` (LM Studio / vLLM / SGLang / llama.cpp all return them);
3. temperature-scales the logits (softmax) for calibrated posterior probabilities;
4. abstains (`value="UNKNOWN"`) when the winner's confidence falls below
   `JEV_ABSTAIN_THRESHOLD` (`auto` scales it to `1.25 / N`), preserving the raw
   argmax in `tentative_value`.

When the endpoint returns no `logprobs`, the client falls back to a JSON
`scores` pass over a full completion, so the service still works on any plain
chat API.

## Install

Requires uv and a running backend (LM Studio by default):

```powershell
uv sync
```

Copy `.env.example` to `.env` and set `JEV_MODEL` to a fast Tier-1/2 model you
have in LM Studio. `JEV_DISABLE_REASONING` defaults to 1 so reasoning models
do not burn the single-token budget in thinking output.

## Usage

### One-shot CLI

```powershell
uv run mcp-agent-openjev doctor

uv run mcp-agent-openjev choice "unauthorized login from an unknown IP" `
    -c billing -c tech_support -c security `
    --criteria "pick the handling department"

uv run mcp-agent-openjev noul "the request is urgent" "this is a support request"

uv run mcp-agent-openjev score "PII exposed in a public bucket for 3 days" `
    -t low -t medium -t high -t critical
```

### HTTP service

```powershell
uv run mcp-agent-openjev http --host 127.0.0.1 --port 8377
curl http://localhost:8377/health
curl -X POST http://localhost:8377/v1/decide/choice -H "Content-Type: application/json" -d '{
  "state": {"ticket": "unauthorized login from an unknown IP"},
  "candidates": ["billing", "tech_support", "security"],
  "criteria": {"billing": "payments", "security": "unauthorized access"}
}'
```

Endpoints: `POST /v1/decide/choice`, `POST /v1/decide/noul`,
`POST /v1/decide/score`, `GET /health`. Choice and score accept optional
`model` / `temperature` / `abstain_threshold` overrides per request.

### MCP server

```powershell
uv run mcp-agent-openjev serve              # stdio (default)
uv run mcp-agent-openjev serve --http       # streamable-http on :8030
```

Tools: `decide_choice`, `decide_noul`, `decide_score`, `decision_status`.
Wire it into opencode's MCP block:

```jsonc
"mcp-agent-openjev": {
  "type": "local",
   "command": ["uv", "--project", "<path-to-mcp_agent_openjev>", "run", "python", "-m", "mcp_agent_openjev", "serve"],
  "environment": {}
}
```

### Examples

```powershell
uv run python examples/ticket_router.py
uv run python examples/severity_score.py
```

### Benchmark harness

OpenJev ships `OpenJevProHarness` — a resumable, concurrent evaluation
benchmark across pluggable engines. Its stock engines (`OpenJevProEngine`,
`DirectStructuredEngine`) only reach Ollama/legacy endpoints, so
`examples/benchmark_harness.py` wires our chat-logprobs client in via the
`OpenJevRouterEngine` adapter:

```powershell
uv run python examples/benchmark_harness.py
```

It runs a small labeled dataset through the harness and prints an
accuracy/latency summary (checkpointed to `harness_checkpoint.json`, resumable
on re-run). Add more engines (e.g. `TypeSafeJevEngine` with a commercial key,
`DirectStructuredEngine` against an Ollama box) to get comparative columns.

## Decisions

| Type | Purpose | Returns |
|---|---|---|
| `Choice` | categorical decision over candidates | `value`, `probabilities`, `confidence`, `abstained`, `tentative_value` |
| `Noul` | binary truth judgment | `value`, `probability_true`, `confidence` |
| `Score` | ordered tier evaluation | `expected_score`, `level_probabilities`, `confidence` |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `JEV_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible base URL |
| `JEV_MODEL` | `google/gemma-4-26b-a4b-qat` | Model id on the backend |
| `JEV_API_KEY` | *(empty)* | Sent as `Authorization: Bearer` only when set; falls back to `LM_STUDIO_API_KEY` |
| `JEV_BACKEND` | `auto` | `chat` (logprobs/scores over chat completions), `ollama` / `legacy` (stock OpenJev clients), or `auto` (ollama only for `:11434`) |
| `JEV_METHOD` | `auto` | `logprobs` (single-token letter logits), `scores` (full JSON likelihood), or `auto` (logprobs with scores fallback) |
| `JEV_TEMPERATURE` | `1.3` | Softmax temperature scaling |
| `JEV_ABSTAIN_THRESHOLD` | `0.45` | Abstention threshold; or `auto` (`1.25 / N`) |
| `JEV_DISABLE_REASONING` | `1` | Inject `reasoning_effort: none` |
| `JEV_TIMEOUT` | `60` | Backend request timeout (seconds) |
| `JEV_TOP_LOGPROBS` | `20` | `top_logprobs` requested for scoring |

## License

This repository (the adapter, service, CLI, MCP surface) is **MIT** (see
`LICENSE`). The `openjevpro` dependency is **PolyForm Noncommercial 1.0.0** —
commercial use of the decision core requires a commercial license from its
maintainers. The full license text is shipped as
`PolyForm NonCommercial 1.0.0.txt` in the repo root for attribution.