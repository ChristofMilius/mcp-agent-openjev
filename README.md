# mcp-agent-openjev

Local **typed probabilistic decision service** — `Choice`, `Noul`, `Score` with
calibrated probabilities and abstention — powered by
[OpenJev](https://huggingface.co/openjev/openjev) (the tuned 27B Qwen3.5 decision
model) over Bionic's **Open Responses** endpoint in
[LM Studio](https://lmstudio.ai).

OpenJev is tuned so a single output position carries the decision: one score per
option letter, then a calibration step. LM Studio's `/v1/completions` and
`/v1/chat/completions` never return logprobs (and cap `top_logprobs` at 20), so
this project drives the one surface that does — `/v1/responses` — with the two
settings that make the readout exact: `reasoning.effort: "none"` (the model emits
exactly one token, the answer letter) and undistorted logits
(`temperature: 1.0`, `top_p: 1.0`, `frequency_penalty: 0`, `presence_penalty: 0`).

For whom: anyone with a local OpenJev GGUF in LM Studio who wants System-1 style
typed decisions (routing, guardrails, intent detection, severity ratings) — and
agents that want them as an MCP tool.

## How it works

For a `Choice` over N options, the client:

1. labels the options `A`, `B`, `C`, … and builds the OpenJev prompt
   (`State:` / `Question:` / `Options:`);
2. sends it to `POST /v1/responses` with `max_tokens=1`, `reasoning.effort=none`,
   and an uncapped `top_logprobs=64`;
3. reads each letter's log-probability at the answer position;
4. temperature-scales the logits (softmax, `T=0.85`) for calibrated posterior
   probabilities;
5. abstains (`value="UNKNOWN"`) when the winner's confidence falls below
   `JEV_ABSTAIN_THRESHOLD` (`auto` scales it to `1.25 / N`), preserving the raw
   argmax in `tentative_value`.

## Install

Requires uv and a running LM Studio with the OpenJev model loaded:

```powershell
uv sync
```

Copy `.env.example` to `.env` and set `JEV_MODEL` to the OpenJev model key LM
Studio serves (default `openjev`). Point `JEV_BASE_URL` at the LM Studio server
(default `http://127.0.0.1:1234`).

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

#### Score tiers

A tier is either a label string or a dict with `label` (or `value`) and an
optional `score`:

```jsonc
// plain labels - weight is the position in the list
["low", "medium", "high", "critical"]

// explicit form - 'score' must equal the tier's position
[{"label": "low", "score": 0}, {"label": "critical", "score": 3}]
```

`score` must match the tier's position because a tier's weight *is* its ordinal
position - that is what makes `expected_score` an expected tier index, which is
what callers compare against. A tier dict without `label`/`value`, a duplicate
label, or a `score` that disagrees with the position is rejected
(`InvalidTierError`, HTTP 400 on the service) rather than silently degraded to a
positional label - a degraded tier is rendered into the prompt as `0. 0`, so the
model ends up ranking meaningless numbers. `level_probabilities` is always keyed
by tier label.

The response also echoes the scale it used, so the score never has to be read
against an assumed convention:

```jsonc
{
  "expected_score": 2.2056,
  "level_probabilities": {"low": 0.00005, "high": 0.783, "critical": 0.211, "UNKNOWN": 0.0002},
  "tier_weights": {"low": 0.0, "high": 2.0, "critical": 3.0, "UNKNOWN": 0.0},
  "confidence": 0.783,
  "abstained": false
}
```

`tier_weights` is ordered by tier, and includes `UNKNOWN` (weight `0.0`) when
abstention is enabled, so `expected_score` can be recomputed from
`level_probabilities` alone.

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

## Decisions

| Type | Purpose | Returns |
|---|---|---|
| `Choice` | categorical decision over candidates | `value`, `probabilities`, `confidence`, `abstained`, `tentative_value` |
| `Noul` | binary truth judgment | `value`, `probability_true`, `confidence` |
| `Score` | ordered tier evaluation | `expected_score`, `level_probabilities`, `confidence` |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `JEV_BASE_URL` | `http://127.0.0.1:1234` | LM Studio server URL |
| `JEV_MODEL` | `openjev` | Model key LM Studio serves |
| `JEV_API_KEY` | *(empty)* | Sent as `Authorization: Bearer` only when set; falls back to `LM_STUDIO_API_KEY` |
| `JEV_TEMPERATURE` | `0.85` | Softmax temperature scaling (OpenJev `READOUT_T`) |
| `JEV_NOUL_T` | `1.829074` | Noul sigmoid scale (OpenJev `READOUT_NOUL_T`) |
| `JEV_NOUL_BIAS` | `0` | Noul sigmoid bias (OpenJev `READOUT_NOUL_BIAS`) |
| `JEV_PERMS` | `1` | Option orders to average over (accuracy mode; costs latency) |
| `JEV_ABSTAIN_THRESHOLD` | `auto` | Abstention threshold; or `auto` (`1.25 / N`) |
| `JEV_TIMEOUT` | `120` | Backend request timeout (seconds) |

The released helper's `READOUT_T`, `READOUT_NOUL_T`, `READOUT_NOUL_BIAS` and
`READOUT_PERMS` variables are also honoured.

## License

The adapter, service, CLI and MCP surface in this repository are **MIT** (see
`LICENSE`). Two upstream licenses are carried alongside because the calibration
mirrors their code and model:

- **`LICENSE-APACHE-2.0.txt`** — Apache 2.0. Covers the `openjev-server` code
  this calibration mirrors (`helper/`, `serve/`) and OpenJev's Apache-2.0 open
  base model, as requested by the [OpenJev model repo](https://huggingface.co/openjev/openjev).
- **CC BY-NC 4.0** — the OpenJev *weights*. Free for research and other
  non-commercial use with attribution; commercial use requires a separate
  licence (open a discussion on the model repo). The terms travel with the
  GGUF, so they apply however the model is served.
