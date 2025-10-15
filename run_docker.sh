#!/bin/bash
set -e  # stop en cas d’erreur

# Nom de l'image
IMAGE_NAME="local-rag-chat"

# Vérifie si le premier argument est --dev
MODE="prod"
if [[ "$1" == "--dev" ]]; then
    MODE="dev"
fi

echo "🧱 Building Docker image (${IMAGE_NAME})..."
docker build -t $IMAGE_NAME .

if [[ "$MODE" == "dev" ]]; then
    echo "🧩 Mode développement activé"
    echo "→ Le code local est monté dans le conteneur (hot reload actif)"
    docker run --rm -it \
        -v "$(pwd):/app" \
        -p 7860:7860 \
        -e LMSTUDIO_HOST=host.docker.internal \
        -e LMSTUDIO_PORT=1234 \
        $IMAGE_NAME \
        uvicorn server:app --host 127.0.0.1 --port 7860 --reload
else
    echo "🚀 Mode production activé"
    docker run --rm -it \
        -v "$(pwd)/pdf:/app/pdf:ro" \
        -v "$(pwd)/vectordb:/app/vectordb" \
        -p 7860:7860 \
        -e LMSTUDIO_HOST=host.docker.internal \
        -e LMSTUDIO_PORT=1234 \
        $IMAGE_NAME
fi
