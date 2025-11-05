# ---------- Base Image ----------
FROM python:3.10-bullseye

# ---------- Environment Settings ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working Directory ----------
WORKDIR /app

# ---------- Copy Project Files ----------
COPY translator /app/

# ---------- Install Dependencies ----------
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---------- Collect Static Files ----------
RUN python manage.py collectstatic --noinput || true

# ---------- Expose Port ----------
EXPOSE 8000

# ---------- Run Using Daphne (for ASGI/WebSocket Support) ----------
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "asl_lite.asgi:application"]
