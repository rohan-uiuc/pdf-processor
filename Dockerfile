FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    pkg-config \
    libpoppler-dev \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    libgl1-mesa-glx \
    postgresql-client \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* && update-ca-certificates

WORKDIR /code

# Set environment variables
ENV TRANSFORMERS_CACHE=/code/.cache/huggingface
ENV HF_HOME=/code/.cache/huggingface
ENV MPLCONFIGDIR=/code/.config/matplotlib
ENV PADDLE_HOME=/code/.paddleocr
ENV HOME=/code
ENV TMPDIR=/code/tmp
ENV PYTHONPATH=/code

RUN mkdir -p /code/data \
    /code/extracted_images \
    /code/.cache \
    /code/.config/matplotlib \
    /code/.paddleocr \
    /code/uploaded_files \
    /code/tmp \
    /code/nltk_data && \
    chown -R 1000:1000 /code && \
    chmod -R 755 /code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the package files first
# COPY processors processors/
# COPY utils utils/
# COPY database.py .
# COPY app.py .
COPY . .

RUN chown -R 1000:1000 /code && \
    chmod -R 755 /code

USER 1000:1000

CMD ["python", "app.py"]