# Agent Runner API Fox Files

Import `AGENT_RUNNER_HTTP_STREAM.postman_collection.json` into API Fox first. It uses absolute localhost URLs and has been shaped like the Postman collection format that API Fox imports reliably.

`AGENT_RUNNER_HTTP_STREAM.openapi.json` is also provided, but the Postman collection is the recommended test artifact because API Fox can be picky about some valid OpenAPI constructs.

Prerequisites:

- Start the external LiteLLM proxy on `http://localhost:4000`.
- Start Agent Runner by running `src/main.py` directly from PyCharm, or with `python .\src\main.py` from `agent-runner/`.
- Use the `smoke-test` agent from `config/agents.json`.

Expected results:

- `Health` returns `{"status":"healthy"}`.
- `Debug Config` shows `lite_llm_base_url` as `http://localhost:4000` and `nacos_enabled` as `false`.
- `Chat Stream - smoke-test agent` returns Server-Sent Events with `Content-Type: text/event-stream`; it is not a JSON response, so API Fox response schema validation should not expect JSON for this endpoint. With a ModelScope token that has chat/inference permission, expect `token_delta` events, an optional real `usage` event when the provider reports token usage, and then `done`. With a token that can list models but cannot call chat completions, expect a real `error` event and then `done`; this still proves the Agent Runner streaming path is not using a mock response.
- The `Visualize` tab merges `token_delta` events into an OpenAI-style non-streaming `chat.completion` JSON. `usage` is filled only from a real upstream usage event; if the provider/SDK does not report usage for the stream, the visualized `prompt_tokens`, `completion_tokens`, and `total_tokens` stay `null`.

The Postman collection is preferred because API Fox imported the earlier demo Postman file reliably, while the OpenAPI import path can be stricter around streaming examples and server metadata.
