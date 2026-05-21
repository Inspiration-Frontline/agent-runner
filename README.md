# Agent Runner

Agent Runner is the request-scoped runtime host for AgentBreaker. It exposes a FastAPI web server and calls models through a local LiteLLM proxy.

## Runtime Configuration

Configuration lives under `config/`:

- `config/litellm.yaml` is the LiteLLM proxy configuration. Mount this file into the LiteLLM container.
- `config/litellm.env` contains the ModelScope and LiteLLM proxy secrets used by Docker.
- `config/agent-runner.env` contains Agent Runner settings.
- `config/agents.json` contains a local smoke-test agent used before `agent-configuration-center` is implemented.

ModelScope connection values:

```text
MODEL_SCOPE_BASE_URL=https://api-inference.modelscope.cn/v1
MODEL_SCOPE_API_KEY=ms-367ff0af-6172-4bfe-a0f4-82dcf0140409
```

The local smoke-test agent uses `Qwen/Qwen3-4B` through LiteLLM. It also has a `mock_response` in `config/agents.json`; Agent Runner tries the real model first and only uses that response if the upstream chat call fails. With the token currently in `config/litellm.env`, ModelScope allows model listing but returns `401` for chat completions, so the fallback keeps the local server smoke-testable until the token is authorized for inference.

## Install

Run from `agent-runner/`:

```powershell
python -m pip install -U .
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

## Start Agent Runner

Run from `agent-runner/`:

```powershell
python -m uvicorn main:app --app-dir src --host 127.0.0.1 --port 8000
```

## Smoke Test

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
