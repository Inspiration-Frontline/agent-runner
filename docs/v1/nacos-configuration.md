# Agent Runner Nacos Configuration

## Overview

Agent Runner supports Nacos as a configuration center, allowing dynamic configuration management
without restarting the service. Configuration can be loaded from both local environment files
and Nacos, with Nacos values taking priority when both are present.

## Configuration Priority

1. **Nacos Configuration** - Highest priority, overrides local values
2. **Local Environment Files** - Fallback when Nacos is unavailable or disabled
3. **Default Values** - Used when neither Nacos nor local config has the value

## Nacos Setup

### Local Development Nacos

- **Console URL**: `http://localhost:8070`
- **Console Username/Password**: `nacos` / `nacos`
- **Client Address**: `127.0.0.1:8848`
- **Namespace**: `agent-breaker-local`

### Configuration Location

Agent Runner loads configuration from:
- **Data ID**: `agent-runner.yaml`
- **Group**: `DEFAULT_GROUP`
- **Namespace**: `agent-breaker-local`

## Nacos Configuration Example

Create a configuration file named `agent-runner.yaml` in Nacos with the following structure:

```yaml
app_name: agent-runner
debug: false

server:
  host: 0.0.0.0
  port: 8000

lite_llm:
  base_url: http://localhost:4000
  api_key: sk-agent-breaker-local
  request_timeout_seconds: 120
  max_retries: 0

services:
  agent_config_center_url: http://localhost:8081
  conversation_service_url: http://localhost:8082
  user_profiler_url: http://localhost:8083
  knowledge_service_url: http://localhost:8084

local_agent_config:
  enabled: true
  path: ./config/agents.json

context:
  max_context_tokens: 128000
  max_output_tokens: 4096

redis:
  host: localhost
  port: 6379
  password: ""
  db: 0
  socket_connect_timeout_seconds: 1
  socket_timeout_seconds: 1

agent_config_cache:
  enabled: false
  ttl_seconds: 300
```

## Environment Variables

Nacos connection can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NACOS_ENABLED` | Enable/disable Nacos configuration | `false` |
| `NACOS_SERVER_ADDRESS` | Nacos server address | `127.0.0.1:8848` |
| `NACOS_NAMESPACE` | Nacos namespace | `agent-breaker-local` |
| `NACOS_DATA_ID` | Configuration Data ID | `agent-runner.yaml` |
| `NACOS_GROUP` | Configuration group | `DEFAULT_GROUP` |
| `NACOS_USERNAME` | Nacos username | `nacos` |
| `NACOS_PASSWORD` | Nacos password | `nacos` |

## Dynamic Configuration Refresh

Agent Runner automatically listens for configuration changes in Nacos. When a configuration
is updated in Nacos, the service will:

1. Receive the configuration change event
2. Parse the new YAML configuration
3. Merge the new values with existing settings
4. Apply the updated configuration immediately

No service restart is required for configuration changes to take effect.

## Enabling Nacos

Local development uses `config/agent-runner.env` with Nacos disabled by default so Agent Runner can start directly from PyCharm. To enable Nacos, set:

```bash
NACOS_ENABLED=true
```

or in `config/agent-runner.env`:

```
NACOS_ENABLED=true
```

## Service URL Configuration

Service URLs for downstream dependencies can be configured in Nacos:

```yaml
services:
  agent_config_center_url: http://agent-configuration-center:8081
  conversation_service_url: http://conversation-manager:8082
  user_profiler_url: http://user-profiler:8083
  knowledge_service_url: http://knowledge-manager:8084
```

This allows easy switching between development, staging, and production environments
by changing the Nacos configuration without modifying local files.

## Redis Cache TTL Configuration

The TTL for agent configuration caching in Redis can be configured via Nacos:

```yaml
agent_config_cache:
  enabled: true
  ttl_seconds: 300
```

This controls how long agent configurations are cached in Redis before being refreshed
from the configuration center.

## Configuration Merge Behavior

When merging configurations:
- Nested configuration in Nacos is flattened to match Settings field names
- Only fields present in Nacos configuration are overridden
- Missing fields retain their values from local configuration or defaults
- Configuration changes are applied atomically

## Testing Configuration Changes

To verify Nacos configuration is working:

1. Start Nacos server locally
2. Publish `agent-runner.yaml` configuration in Nacos console
3. Start agent-runner service
4. Check logs for "Configuration merged from Nacos and local files"
5. Modify a value in Nacos console
6. Check logs for "Configuration cache updated from Nacos"
7. Verify the new value is used by the service

## Troubleshooting

### Nacos Connection Failed

If you see warnings like "Failed to initialize Nacos client", check:
- Nacos server is running and accessible
- Server address, username, and password are correct
- Namespace exists in Nacos
- Network connectivity to Nacos server

### Configuration Not Loading

If Nacos configuration is not being applied:
- Check Data ID matches exactly (including file extension `.yaml`)
- Verify the configuration group is `DEFAULT_GROUP`
- Ensure namespace ID is correct (not namespace name)
- Check YAML syntax is valid

### Using Local Configuration Only

If you want to bypass Nacos temporarily:
- Set `NACOS_ENABLED=false` in environment
- Or delete/comment out the Nacos configuration in Nacos console
- Local configuration files will be used as fallback
