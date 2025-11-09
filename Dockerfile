# ---------- Base Image ----------
FROM python:3.10-slim-bullseye

# ---------- Environment Settings ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ---------- Working Directory ----------
WORKDIR /app

# ---------- Install System Dependencies ----------
# Only lightweight dependencies needed for Django, Channels, Daphne
RUN apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---------- Copy and Install Dependencies ----------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------- Copy Project Files ----------
COPY . .

# ---------- Collect Static Files ----------
RUN python manage.py collectstatic --noinput || true

# ---------- Expose Port ----------
EXPOSE 8000

# ---------- Run Using Daphne (for ASGI/WebSocket Support) ----------
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "asl_lite.asgi:application"]
