# 🚀 Fullstack RAG Video Platform - Complete Setup Guide

This guide will walk you through setting up and deploying the RAG Video Platform from scratch.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- **Python 3.9-3.13** - Backend runtime
- **Node.js 18+** - Frontend runtime
- **Docker & Docker Compose** (for containerized deployment)
- **Git** - Version control

### Required API Keys

1. **OpenAI API Key** - For LLM and embeddings
   - Get from: https://platform.openai.com/api-keys

2. **Deepgram API Key** - For speech-to-text
   - Get from: https://console.deepgram.com/

3. **ElevenLabs API Key** - For text-to-speech
   - Get from: https://elevenlabs.io/app/settings/api-keys

### Optional API Keys

- **Anthropic API Key** - Alternative LLM provider
- **Google API Key** - Alternative LLM/STT provider
- **Qdrant Cloud API Key** - For cloud vector database
- **Avatar Provider Keys** - Simli, Tavus, Hedra, etc.

## Local Development Setup

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd agents/examples/fullstack-rag-video-platform
```

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

**Important**: Update the following in `.env`:
```env
OPENAI_API_KEY=sk-your-actual-openai-key
DEEPGRAM_API_KEY=your-actual-deepgram-key
ELEVENLABS_API_KEY=your-actual-elevenlabs-key
```

### Step 3: Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Copy environment template
cp .env.example .env.local

# Edit .env.local
nano .env.local
```

Update `.env.local`:
```env
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

### Step 4: Start LiveKit Server

Option A - Using Docker (Recommended):
```bash
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  -e LIVEKIT_KEYS="devkey: secret" \
  livekit/livekit-server
```

Option B - Download binary:
```bash
# Download from https://github.com/livekit/livekit/releases
# Extract and run:
./livekit-server --dev
```

### Step 5: Start Backend Services

Open a new terminal:

```bash
cd backend
source venv/bin/activate

# Start API server
python api_server.py
```

Open another terminal:

```bash
cd backend
source venv/bin/activate

# Start agent
python agent.py dev
```

### Step 6: Start Frontend

Open a new terminal:

```bash
cd frontend
npm run dev
```

### Step 7: Access the Application

Open your browser and navigate to:
- Frontend: http://localhost:3000
- API Server: http://localhost:8000
- API Docs: http://localhost:8000/docs
- LiveKit Server: http://localhost:7880

## Docker Deployment

### Quick Start with Docker Compose

```bash
# From the project root
cd fullstack-rag-video-platform

# Create .env file for Docker Compose
cat > .env << EOF
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
OPENAI_API_KEY=your-openai-key
DEEPGRAM_API_KEY=your-deepgram-key
ELEVENLABS_API_KEY=your-elevenlabs-key
EOF

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Service URLs (Docker)

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- LiveKit: ws://localhost:7880
- Qdrant: http://localhost:6333

### Verify Deployment

```bash
# Check service health
docker-compose ps

# Check backend health
curl http://localhost:8000/health

# Check Qdrant
curl http://localhost:6333/collections
```

## Production Deployment

### AWS Deployment

#### Prerequisites
- AWS Account with ECS/EKS access
- RDS PostgreSQL instance (optional, for production memory)
- S3 bucket for document storage
- CloudFront distribution (optional, for CDN)

#### Steps

1. **Build and Push Docker Images**

```bash
# Login to AWS ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push backend
cd backend
docker build -t rag-video-backend .
docker tag rag-video-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-video-backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-video-backend:latest

# Build and push frontend
cd ../frontend
docker build -t rag-video-frontend .
docker tag rag-video-frontend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-video-frontend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-video-frontend:latest
```

2. **Deploy to ECS**

- Create ECS cluster
- Create task definitions for backend and frontend
- Configure environment variables
- Set up Application Load Balancer
- Deploy services

3. **Configure LiveKit**

For production, use LiveKit Cloud:
- Sign up at https://cloud.livekit.io
- Create a project
- Get production API keys
- Update environment variables

### GCP Deployment

#### Using Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/<project-id>/rag-video-backend backend/
gcloud builds submit --tag gcr.io/<project-id>/rag-video-frontend frontend/

# Deploy backend
gcloud run deploy rag-video-backend \
  --image gcr.io/<project-id>/rag-video-backend \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY

# Deploy frontend
gcloud run deploy rag-video-frontend \
  --image gcr.io/<project-id>/rag-video-frontend \
  --platform managed \
  --region us-central1
```

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace rag-video

# Create secrets
kubectl create secret generic api-keys \
  --from-literal=openai-key=$OPENAI_API_KEY \
  --from-literal=deepgram-key=$DEEPGRAM_API_KEY \
  --from-literal=elevenlabs-key=$ELEVENLABS_API_KEY \
  -n rag-video

# Deploy (create k8s manifests first)
kubectl apply -f k8s/ -n rag-video

# Check deployment
kubectl get pods -n rag-video
```

## Configuration

### Backend Configuration

All backend settings are in `backend/config.py` and can be overridden with environment variables:

#### LLM Configuration
```env
LLM_PROVIDER=openai  # openai, anthropic, google, groq
LLM_MODEL=gpt-4-turbo
LLM_TEMPERATURE=0.7
```

#### RAG Configuration
```env
EMBEDDING_MODEL=text-embedding-3-large
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
```

#### Vector Database
```env
VECTOR_DB_TYPE=qdrant  # local, qdrant, pinecone
VECTOR_DB_URL=http://localhost:6333
```

#### Video Configuration
```env
ENABLE_VIDEO=true
AVATAR_PROVIDER=simli  # simli, tavus, hedra, etc.
VIDEO_FPS=30
VIDEO_QUALITY=high
```

### Frontend Configuration

Configure in `.env.local`:

```env
# LiveKit connection
NEXT_PUBLIC_LIVEKIT_URL=wss://your-livekit-server.com

# LiveKit credentials (server-side only)
LIVEKIT_API_KEY=your-key
LIVEKIT_API_SECRET=your-secret

# Backend API (optional)
NEXT_PUBLIC_API_URL=https://api.your-domain.com
```

## Advanced Features

### Using Different Vector Databases

#### Pinecone

```bash
pip install pinecone-client

# Set environment
export VECTOR_DB_TYPE=pinecone
export PINECONE_API_KEY=your-key
export PINECONE_ENVIRONMENT=us-east-1-aws
```

#### Weaviate

```bash
pip install weaviate-client

# Set environment
export VECTOR_DB_TYPE=weaviate
export WEAVIATE_URL=http://localhost:8080
```

### Custom Avatar Integration

To add a custom avatar provider, edit `backend/video_handler.py`:

```python
class VideoHandler:
    def __init__(self, avatar_provider: str = "custom"):
        if avatar_provider == "custom":
            # Initialize your custom avatar SDK
            pass
```

### Multi-Agent Setup

For multiple specialized agents, create separate agent files:

```python
# backend/agents/support_agent.py
async def support_agent_entrypoint(ctx: JobContext):
    # Specialized support agent
    pass

# backend/agents/sales_agent.py
async def sales_agent_entrypoint(ctx: JobContext):
    # Specialized sales agent
    pass
```

## Monitoring and Analytics

### Enable Prometheus Metrics

```env
ENABLE_METRICS=true
METRICS_PORT=9090
```

Access metrics at: http://localhost:9090/metrics

### Enable OpenTelemetry

```env
ENABLE_TELEMETRY=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

### View Logs

```bash
# Docker logs
docker-compose logs -f backend

# Application logs
tail -f backend/logs/app.log
```

## Troubleshooting

### Common Issues

#### 1. LiveKit Connection Failed

**Symptom**: Frontend can't connect to video session

**Solutions**:
- Verify LiveKit server is running: `curl http://localhost:7880`
- Check `NEXT_PUBLIC_LIVEKIT_URL` matches server URL
- Ensure API keys match between frontend and LiveKit server

#### 2. RAG Queries Not Working

**Symptom**: Agent can't retrieve documents

**Solutions**:
- Check if documents are uploaded: http://localhost:8000/api/documents
- Verify vector database is running: `curl http://localhost:6333`
- Check OpenAI API key is valid
- Review logs: `docker-compose logs backend`

#### 3. Audio Not Working

**Symptom**: No audio from agent

**Solutions**:
- Verify Deepgram API key is set
- Check ElevenLabs API key is valid
- Ensure microphone permissions are granted in browser
- Test STT/TTS individually

#### 4. High Memory Usage

**Symptom**: Backend consuming too much memory

**Solutions**:
- Reduce `CHUNK_SIZE` to 256
- Lower `MAX_CONCURRENT_SESSIONS`
- Enable caching: `ENABLE_CACHING=true`
- Use external vector database instead of local

#### 5. Slow Response Times

**Symptom**: Agent responses are slow

**Solutions**:
- Use faster LLM model: `gpt-4o-mini` or `gpt-3.5-turbo`
- Reduce `TOP_K` to 3
- Enable Redis caching
- Use streaming responses

### Debug Mode

Enable detailed logging:

```env
# Backend
LOG_LEVEL=DEBUG

# Frontend
NEXT_PUBLIC_DEBUG=true
```

### Getting Help

- **Documentation**: https://docs.livekit.io
- **Discord**: https://livekit.io/discord
- **GitHub Issues**: Create an issue with:
  - Error logs
  - Environment details
  - Steps to reproduce

## Performance Optimization

### Backend Optimization

1. **Use Redis for caching**
```env
REDIS_URL=redis://localhost:6379
ENABLE_CACHING=true
```

2. **Enable connection pooling**
```env
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

3. **Optimize vector search**
```env
TOP_K=3  # Reduce from 5
CHUNK_SIZE=256  # Reduce from 512
```

### Frontend Optimization

1. **Enable CDN for static assets**
2. **Use Next.js Image optimization**
3. **Implement code splitting**
4. **Enable service worker caching**

## Security Best Practices

1. **Never commit API keys** - Use environment variables
2. **Enable HTTPS** in production
3. **Implement rate limiting**
4. **Use JWT authentication** for API endpoints
5. **Sanitize user inputs**
6. **Regular security updates**

```env
# Enable authentication
ENABLE_AUTH=true
JWT_SECRET=your-super-secret-key-change-in-production
```

## Scaling

### Horizontal Scaling

Use Kubernetes or ECS with auto-scaling:

```yaml
# k8s/deployment.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-backend
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Database Scaling

For production, use managed services:
- **RDS/Cloud SQL** for PostgreSQL
- **Qdrant Cloud** for vector database
- **Redis Cloud** for caching

## Backup and Recovery

### Backup Strategy

```bash
# Backup vector database
docker exec qdrant tar czf /qdrant/storage/backup.tar.gz /qdrant/storage

# Backup conversation memory
sqlite3 data/memory.db ".backup 'backup/memory_$(date +%Y%m%d).db'"

# Backup documents
tar czf backup/documents_$(date +%Y%m%d).tar.gz storage/documents/
```

### Automated Backups

Create a cron job:
```bash
# /etc/cron.daily/rag-backup
#!/bin/bash
cd /path/to/fullstack-rag-video-platform
./scripts/backup.sh
```

## Next Steps

1. **Customize the agent** - Edit prompts in `backend/agent.py`
2. **Add custom tools** - Extend functionality in `backend/tools.py`
3. **Integrate analytics** - Connect to your analytics platform
4. **Add authentication** - Implement user management
5. **Customize UI** - Modify frontend components

## Support

For questions or issues:
- Check the main [README.md](./README.md)
- Review [LiveKit documentation](https://docs.livekit.io)
- Join [LiveKit Discord](https://livekit.io/discord)

---

**Congratulations!** You now have a fully functional RAG-powered video platform running. 🎉
