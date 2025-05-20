# STAGE 1: Builder Stage
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR ${APP_HOME}

# Install system dependencies (including libgl1 instead of libgl1-mesa-glx)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --prefix=/install --default-timeout=300 --retries 5

# Copy source code
COPY ./API/. ${APP_HOME}/API/
COPY ./Data_Pipeline/. ${APP_HOME}/Data_Pipeline/
COPY ./Model/. ${APP_HOME}/Model/

# STAGE 2: Final Runtime Image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV PATH="/install/bin:$PATH"

WORKDIR ${APP_HOME}

# Reinstall libgl1 for OpenCV runtime support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=builder /install/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /install/bin /install/bin

# Copy app code
COPY --from=builder ${APP_HOME}/API/ ${APP_HOME}/API/
COPY --from=builder ${APP_HOME}/Data_Pipeline/ ${APP_HOME}/Data_Pipeline/
COPY --from=builder ${APP_HOME}/Model/ ${APP_HOME}/Model/

EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "API.apiExperimental:app", "--bind", "0.0.0.0:8000", "--timeout", "300"]
