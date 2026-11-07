FROM python:3.11-slim

# opencv-contrib-python (pulled in by PaddleOCR) needs libGL/glib; PaddlePaddle needs libgomp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/ocr
COPY pyproject.toml README.md requirements-lock.txt ./
COPY app ./app
RUN pip install --no-cache-dir -c requirements-lock.txt .

ENV OCR_MODEL_CACHE_DIR=/models
VOLUME /models
EXPOSE 8000

RUN useradd --create-home ocr && mkdir -p /models && chown ocr /models
USER ocr

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
