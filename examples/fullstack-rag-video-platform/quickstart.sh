#!/bin/bash

# Fullstack RAG Video Platform - Quick Start Script
# This script helps you get started quickly with the platform

set -e

echo "🚀 Fullstack RAG Video Platform - Quick Start"
echo "=============================================="
echo ""

# Check for required commands
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed. Aborting." >&2; exit 1; }

echo "✓ Docker and Docker Compose are installed"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cat > .env << EOF
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
OPENAI_API_KEY=
DEEPGRAM_API_KEY=
ELEVENLABS_API_KEY=
EOF
    echo "⚠️  Please edit .env and add your API keys:"
    echo "   - OPENAI_API_KEY (required)"
    echo "   - DEEPGRAM_API_KEY (required)"
    echo "   - ELEVENLABS_API_KEY (required)"
    echo ""
    read -p "Press Enter after adding your API keys to .env..."
fi

# Verify API keys are set
source .env
if [ -z "$OPENAI_API_KEY" ] || [ -z "$DEEPGRAM_API_KEY" ] || [ -z "$ELEVENLABS_API_KEY" ]; then
    echo "❌ Required API keys are missing in .env file"
    exit 1
fi

echo "✓ API keys configured"
echo ""

# Start services
echo "🐳 Starting Docker services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo ""
echo "🔍 Checking service health..."

if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ Backend API is healthy"
else
    echo "⚠️  Backend API is not responding yet (this is normal, give it a minute)"
fi

if curl -s http://localhost:3000 > /dev/null; then
    echo "✓ Frontend is healthy"
else
    echo "⚠️  Frontend is not responding yet (this is normal, give it a minute)"
fi

echo ""
echo "=============================================="
echo "🎉 Platform is starting up!"
echo ""
echo "📍 Access the platform at:"
echo "   Frontend:  http://localhost:3000"
echo "   API Docs:  http://localhost:8000/docs"
echo "   Qdrant:    http://localhost:6333/dashboard"
echo ""
echo "📚 View logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Stop services:"
echo "   docker-compose down"
echo ""
echo "📖 For detailed setup instructions, see SETUP_GUIDE.md"
echo "=============================================="
