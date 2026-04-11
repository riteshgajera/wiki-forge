FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF + Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." 2>&1

COPY . .

# Create runtime dirs
RUN mkdir -p raw wiki data/vectors

EXPOSE 8000

CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
