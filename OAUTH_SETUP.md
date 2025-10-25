# OAuth Authentication Setup Guide

## ✅ OAuth Provider Working!

OAuth Provider с Dynamic Client Registration (DCR) работает корректно.

### 📋 Что требуется для работы OAuth:

#### 1. Environment Variables

```bash
# Включить OAuth
USE_OAUTH2=true

# OAuth Issuer URL (ОБЯЗАТЕЛЬНО HTTPS в production!)
OAUTH2_ISSUER=https://your-domain.com

# OAuth Secret Key (для подписи токенов)
OAUTH2_SECRET_KEY=your-secret-key-min-32-chars

# OAuth Scopes (опционально, по умолчанию: read:data,write:data)
OAUTH2_SCOPES=read:data,write:data,admin

# Required Scopes (опционально, по умолчанию: read:data)
OAUTH2_REQUIRED_SCOPES=read:data
```

#### 2. HTTPS Requirement

**⚠️ ВАЖНО:** В production OAuth **ОБЯЗАТЕЛЬНО** требует HTTPS!

- Используйте reverse proxy (nginx, Caddy, Traefik)
- Настройте SSL/TLS сертификаты
- `OAUTH2_ISSUER` должен быть `https://`

#### 3. Claude Web Callback URL

Claude Web использует callback URL:
```
https://claude.ai/api/mcp/auth_callback
```

Этот URL автоматически разрешен в FastMCP OAuth Provider.

### 🔧 Доступные OAuth Endpoints

После запуска с `USE_OAUTH2=true`:

```
/.well-known/oauth-authorization-server  - OAuth Server Metadata
/.well-known/oauth-protected-resource    - Protected Resource Metadata
/register                                - Dynamic Client Registration (DCR)
/authorize                               - Authorization Endpoint
/token                                   - Token Endpoint
/revoke                                  - Token Revocation Endpoint
```

### 🎯 Подключение к Claude Web

1. **Запустите сервер с OAuth:**
   ```bash
   USE_OAUTH2=true OAUTH2_ISSUER=https://your-domain.com docker-compose up
   ```

2. **В Claude Web:**
   - Перейдите в [Settings > Connectors](https://claude.ai/settings/connectors)
   - Нажмите "Add custom connector"
   - Введите URL: `https://your-domain.com`
   - **НЕ указывайте** Client ID/Secret (DCR сделает это автоматически!)
   - Нажмите "Add"

3. **Авторизуйтесь:**
   - В чате нажмите "Search and tools"
   - Найдите ваш connector
   - Нажмите "Connect"
   - Пройдите OAuth flow в браузере

### 🔐 Режимы Аутентификации

Сервер поддерживает 3 режима:

1. **OAuth Provider** (`USE_OAUTH2=true`)
   - Полный OAuth 2.1 сервер с DCR
   - Для Claude Web custom connectors
   - Требует HTTPS в production

2. **API Key** (`MCP_API_KEY=your-key`)
   - Простая Bearer token аутентификация
   - Для внутренних сервисов
   - Не требует HTTPS (но рекомендуется)

3. **No Auth** (оба выключены)
   - Открытый сервер без аутентификации
   - Только для разработки/тестирования

### ✅ Проверка работы

```bash
# Проверить OAuth metadata
curl https://your-domain.com/.well-known/oauth-authorization-server

# Проверить DCR
curl -X POST https://your-domain.com/register \
  -H "Content-Type: application/json" \
  -d '{"redirect_uris": ["https://claude.ai/api/mcp/auth_callback"]}'
```

### 📝 Примечания

- OAuth Provider автоматически создает все необходимые endpoints
- DCR (Dynamic Client Registration) включен по умолчанию
- PKCE (Proof Key for Code Exchange) поддерживается
- Token refresh и revocation работают
- Все требования MCP спецификации выполнены
