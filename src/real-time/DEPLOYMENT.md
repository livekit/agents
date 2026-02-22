# Real-Time Vision + Avatar Agent - Deployment Guide

This guide covers deploying the real-time agent and frontend to production.

## Prerequisites

- Self-hosted LiveKit server (with database and egress configured)
- Python 3.9+ hosting environment
- Node.js 18+ hosting environment
- API keys for: OpenAI, Deepgram, ElevenLabs, Anam

## Backend Deployment

### Option 1: Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY agent-py/pyproject.toml .
COPY agent-py/src ./src

RUN pip install --no-cache-dir -e "."

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/agent.py", "start"]
```

**Build and run:**
```bash
docker build -t realtime-agent .
docker run -d \
  -e LIVEKIT_URL=ws://your-livekit-server:7880 \
  -e LIVEKIT_API_KEY=your-api-key \
  -e LIVEKIT_API_SECRET=your-api-secret \
  -e OPENAI_API_KEY=sk-... \
  -e DEEPGRAM_API_KEY=... \
  -e ELEVENLABS_API_KEY=... \
  -e ANAM_API_KEY=... \
  -e ANAM_AVATAR_ID=... \
  realtime-agent
```

### Option 2: Systemd Service

**Create `/etc/systemd/system/realtime-agent.service`:**
```ini
[Unit]
Description=Real-Time Agent
After=network.target

[Service]
Type=simple
User=livekit
WorkingDirectory=/opt/realtime-agent
Environment="PYTHONUNBUFFERED=1"
Environment="LIVEKIT_URL=ws://localhost:7880"
EnvironmentFile=/etc/realtime-agent/.env
ExecStart=/opt/realtime-agent/venv/bin/python src/agent.py start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl enable realtime-agent
sudo systemctl start realtime-agent
sudo systemctl status realtime-agent
```

### Option 3: Kubernetes

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: realtime-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: realtime-agent
  template:
    metadata:
      labels:
        app: realtime-agent
    spec:
      containers:
      - name: agent
        image: realtime-agent:latest
        imagePullPolicy: Always
        env:
        - name: LIVEKIT_URL
          value: "wss://livekit.example.com"
        - name: LIVEKIT_API_KEY
          valueFrom:
            secretKeyRef:
              name: livekit-secrets
              key: api-key
        - name: LIVEKIT_API_SECRET
          valueFrom:
            secretKeyRef:
              name: livekit-secrets
              key: api-secret
        envFrom:
        - secretRef:
            name: api-keys
        resources:
          requests:
            memory: "256Mi"
            cpu: "500m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
```

## Frontend Deployment

### Option 1: Vercel (Recommended for Next.js)

**Install Vercel CLI:**
```bash
npm install -g vercel
```

**Deploy:**
```bash
cd frontend
vercel --env-file=.env.local
```

**Environment variables** (set in Vercel dashboard):
```
NEXT_PUBLIC_LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret
```

### Option 2: Docker + Cloud Run (GCP)

**Dockerfile:**
```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend . .
RUN npm run build

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package*.json ./
RUN npm install --production

ENV PORT 3000
EXPOSE 3000

CMD ["npm", "start"]
```

**Build and deploy:**
```bash
docker build -t gcr.io/your-project/realtime-frontend:latest .
docker push gcr.io/your-project/realtime-frontend:latest
gcloud run deploy realtime-frontend \
  --image gcr.io/your-project/realtime-frontend:latest \
  --platform managed \
  --region us-central1 \
  --set-env-vars NEXT_PUBLIC_LIVEKIT_URL=wss://your-livekit-server.com
```

### Option 3: AWS Amplify

**Prerequisites:**
- AWS account with GitHub integration

**Steps:**
1. Push code to GitHub
2. Connect to AWS Amplify
3. Set environment variables in Amplify console
4. Deploy

**amplify.yml:**
```yaml
version: 1
frontend:
  phases:
    preBuild:
      commands:
        - npm install
    build:
      commands:
        - npm run build
  artifacts:
    baseDirectory: .next
    files:
      - '**/*'
  cache:
    paths:
      - node_modules/**/*
```

## LiveKit Server Configuration

Ensure your LiveKit server is properly configured for agents:

**livekit.yaml:**
```yaml
port: 7880
bind_addresses:
  - 0.0.0.0

# Database
db:
  log_level: info

# RTC Configuration
rtc:
  use_external_ip: true
  auto_track_room_leave: true
  packet_loss_feedback_interval: 0

# Webhook (optional, for logging)
webhook:
  api_key: your-webhook-key
  urls:
    - http://your-server/webhooks/livekit

# Keys for agents
keys:
  - key: devkey
    secret: secret

logging:
  level: info
```

## Monitoring & Logging

### Backend Monitoring

**Health check endpoint** (add to agent.py):
```python
@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Metrics to track:**
- Agent session count
- Average response latency
- Error rates
- Avatar initialization time
- Token usage (OpenAI)

### Frontend Monitoring

Use Sentry, LogRocket, or similar:

```typescript
import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: "your-sentry-dsn",
  environment: process.env.NODE_ENV,
  tracesSampleRate: 1.0,
});
```

## Performance Tuning

### Backend

- Enable HTTP/2 for better connection handling
- Use connection pooling for database
- Cache VAD model in memory
- Monitor GPU usage (if using GPU-accelerated TTS)

### Frontend

- Enable image optimization
- Use CDN for static assets
- Enable compression (gzip/brotli)
- Lazy load components

## Security

### HTTPS/WSS

Ensure all connections use TLS:

```bash
# LiveKit server
livekit-server --bind 0.0.0.0 \
  --cert /etc/livekit/cert.pem \
  --key /etc/livekit/key.pem
```

### API Keys

- Use secrets manager (AWS Secrets Manager, HashiCorp Vault)
- Rotate keys regularly
- Use environment-specific keys
- Never commit `.env.local` to version control

### CORS

Configure CORS for frontend:

```typescript
// In Next.js API routes
const allowedOrigins = [
  "https://your-domain.com",
  "https://app.your-domain.com",
];

export async function POST(request: Request) {
  const origin = request.headers.get("origin");
  
  if (!allowedOrigins.includes(origin || "")) {
    return new Response("Forbidden", { status: 403 });
  }
  
  // ... rest of handler
}
```

## Scaling

### Horizontal Scaling

Deploy multiple agent instances behind a load balancer:

```yaml
# Nginx config
upstream agents {
  server agent1:3000;
  server agent2:3000;
  server agent3:3000;
}

server {
  listen 80;
  location / {
    proxy_pass http://agents;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
  }
}
```

### Database Connection Pooling

Use PgBouncer for PostgreSQL (if using LiveKit database):

```ini
[databases]
livekit = host=localhost port=5432 dbname=livekit

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
```

## Disaster Recovery

### Backup LiveKit Data

```bash
# PostgreSQL backup
pg_dump -h localhost -U postgres livekit > backup.sql

# Restore
psql -h localhost -U postgres livekit < backup.sql
```

### Health Checks

Create monitoring script:

```bash
#!/bin/bash
# check-health.sh

BACKEND_URL="https://your-backend.com/health"
FRONTEND_URL="https://your-frontend.com"

# Check backend
if ! curl -f $BACKEND_URL > /dev/null; then
  echo "Backend health check failed"
  exit 1
fi

# Check frontend
if ! curl -f $FRONTEND_URL > /dev/null; then
  echo "Frontend health check failed"
  exit 1
fi

echo "All services healthy"
```

Schedule with cron:
```bash
*/5 * * * * /opt/scripts/check-health.sh
```

## Troubleshooting Deployment

### Backend won't connect to LiveKit

```bash
# Test connectivity
telnet your-livekit-server 7880

# Check WebSocket
wscat -c wss://your-livekit-server/ws
```

### Frontend can't get JWT token

1. Check API route is accessible
2. Verify environment variables are set
3. Check CORS configuration
4. Review API logs for errors

### High latency

1. Check network latency between components
2. Enable preemptive generation
3. Reduce video frame sampling rate
4. Check LiveKit server resource usage

## Maintenance

### Regular Tasks

- Monitor API usage and costs
- Rotate secrets monthly
- Update dependencies
- Review and clean old recordings
- Check disk space on LiveKit server

### Updates

Deploy updates with zero downtime:

```bash
# Backend: Use rolling deployment
kubectl set image deployment/realtime-agent \
  agent=realtime-agent:new-version --record

# Frontend: Vercel auto-deployment on git push
```

## References

- [LiveKit Deployment Guide](https://docs.livekit.io/deploy/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Next.js Deployment](https://nextjs.org/docs/deployment)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/)
