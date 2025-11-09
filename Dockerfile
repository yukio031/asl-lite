# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- Environment ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ---------- Copy requirements separately (for caching) ----------
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---------- Copy rest of project ----------
COPY . .

# ---------- Collect Static (optional) ----------
RUN python manage.py collectstatic --noinput || true

EXPOSE 8000

# ---------- Run with Daphne ----------
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "asl_lite.asgi:application"]
