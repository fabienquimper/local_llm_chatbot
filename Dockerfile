# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Empêcher la création de fichiers .pyc et le buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Installer dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement les fichiers nécessaires
COPY server.py main_rag_chat.py util_*.py ./ 
COPY static ./static
#COPY pdf ./pdf
COPY bases.json ./bases.json
COPY uploads ./uploads
COPY bases ./bases

# Créer le dossier PDF (monté en volume depuis l’hôte)
RUN mkdir -p /app/pdf

# Exposer le port du serveur FastAPI
EXPOSE 7860

# Lancer automatiquement le serveur FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
