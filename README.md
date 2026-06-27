# Agent Runner

Agent Runner is the request-scoped runtime host for AgentBreaker. It exposes a FastAPI web server, runs agents through `openai-agents-python`, uses the SDK LiteLLM model integration in-process, and forwards model traffic to the external LiteLLM proxy.

## Runtime Configuration

Configuration lives under `config/`:

- `config/litellm.yaml` is the LiteLLM proxy configuration. Mount this file into the LiteLLM container.
- `config/litellm.env` contains the ModelScope and LiteLLM proxy secrets used by Docker.
- `config/agent-runner.env` contains Agent Runner settings.
- `config/agents.json` contains a local smoke-test agent used before `agent-configuration-center` is implemented.

Provider connection values are kept in `config/litellm.env`. Use placeholder names in docs and keep real API keys local:

```text
MODEL_SCOPE_BASE_URL=https://api-inference.modelscope.cn/v1
MODEL_SCOPE_API_KEY=<your-modelscope-key>
OWIWO_BASE_URL=https://api.owiwo.cn/v1
OWIWO_API_KEY=<your-owiwo-key>
LITELLM_MASTER_KEY=sk-agent-breaker-local
```

The local smoke-test agent uses `Qwen/Qwen3-235B-A22B-Instruct-2507` through the SDK LiteLLM integration and the external LiteLLM proxy. Local startup reads `config/agent-runner.env` by default, so PyCharm can run `src/main.py` directly without manual environment variables.

## Install

Run from `agent-runner/`:

```powershell
uv sync --no-install-project

# Do not install the project package itself unless that is explicitly needed.
```

## Start LiteLLM

Run from `agent-runner/`:

```powershell
docker rm -f agentbreaker-litellm
docker run -d --name agentbreaker-litellm `
  -p 4000:4000 `
  --env-file .\config\litellm.env `
  -v "${PWD}\config\litellm.yaml:/app/config.yaml" `
  litellm/litellm:latest --config /app/config.yaml --port 4000
```

## Add LiteLLM Models

To add a model to the local LiteLLM proxy:

1. Add provider secrets or base URLs to `config/litellm.env`.
2. Add one entry under `model_list` in `config/litellm.yaml`.
3. Recreate the LiteLLM container with the command from `Start LiteLLM`.

Example:

```yaml
model_list:
  - model_name: gpt-5.5
    litellm_params:
      model: openai/gpt-5.5
      api_base: os.environ/OWIWO_BASE_URL
      api_key: os.environ/OWIWO_API_KEY
      headers:
        User-Agent: Mozilla/5.0
      timeout: 120
```

Notes:

- The Owiwo endpoint needs `https://api.owiwo.cn/v1`; the root `https://api.owiwo.cn/` returns the web console HTML, not the OpenAI-compatible API.
- Owiwo currently blocks the default OpenAI/LiteLLM client user agent, so keep `headers.User-Agent: Mozilla/5.0` for that provider.
- Non-streaming and streaming Owiwo responses both report real `usage` through LiteLLM when `stream_options.include_usage` is requested. Agent Runner only forwards usage when the upstream streaming response provides it; it does not estimate token counts.

## Start Agent Runner

In PyCharm, open `src/main.py` and click Run. The default local configuration disables Nacos and uses the local LiteLLM proxy at `http://localhost:4000`.

The equivalent terminal command is:

```powershell
python .\src\main.py
```

## Smoke Test

After Agent Runner is started:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health

$body = @{
  agent_id = "smoke-test"
  user_id = "local-user"
  message = "Say hello in one short sentence."
} | ConvertTo-Json

Invoke-WebRequest `
  -Method POST `
  -Uri http://127.0.0.1:8000/v1/agent/chat/stream `
  -Body $body `
  -ContentType "application/json"
```

The stream endpoint always exercises the real chain: Agent Runner -> OpenAI Agents SDK -> in-process LiteLLM model integration -> external LiteLLM proxy -> ModelScope. No mock response is used in the local run path.

If `config/litellm.env` contains a ModelScope token that can list models but cannot call chat completions, the stream still returns HTTP 200 SSE with a real `error` event followed by `done`. After replacing `MODEL_SCOPE_API_KEY` with a token that has inference/chat permission and restarting the LiteLLM container, the same request should return `token_delta` events followed by `done`.

For API Fox, import `apifox/AGENT_RUNNER_HTTP_STREAM.postman_collection.json`.
