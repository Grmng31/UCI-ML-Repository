#Dockerfile Imagen base ligera de Python 3.11
FROM python:3.11-slim

# Evita archivos .pyc y muestra logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instalacion de dependencias del sistema (opcional, para compatibilidad con librerias compiladas)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copia del archivo de dependencias e instalacion
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copia del codigo fuente y recursos necesarios
COPY src/ ./src/
COPY tests/ ./tests/
COPY artifacts/ ./artifacts/ 2>/dev/null || true
COPY models/ ./models/ 2>/dev/null || true

# Punto de entrada por defecto para entrenamiento
ENTRYPOINT ["python", "-m", "src.train"]

# Alternativas comentadas para otros comandos:
# Para evaluacion: docker run --rm adult-mlops-multimodel python -m src.evaluate
# Para tests: docker run --rm adult-mlops-multimodel python -m tests.test_models
# Para pipeline completo DVC: docker run --rm adult-mlops-multimodel dvc repro