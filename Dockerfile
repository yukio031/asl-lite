# ---------- Base Image ----------
FROM python:3.10-slim-bullseye

# ---------- Environment Settings ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------- Working Directory ----------
WORKDIR /app

# ---------- Copy Project Files ----------
COPY . .

# ---------- Install System Dependencies ----------
RUN apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---------- Install Python Dependencies ----------
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---------- Collect Static Files ----------
RUN python manage.py collectstatic --noinput || true

# ---------- Expose Port ----------
EXPOSE 8000

# ---------- Run Using Daphne (for ASGI/WebSocket Support) ----------
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "asl_lite.asgi:application"]
