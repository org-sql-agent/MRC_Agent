FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- 系統相依 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    git curl ffmpeg libgl1 ca-certificates \
 && ln -sf /usr/bin/python3.11 /usr/bin/python \
 && python -m pip install --upgrade pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# PyTorch (CUDA 12.1) 官方索引
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1


# 其餘依賴
RUN pip install --no-cache-dir -r /app/requirements.txt

# 再複製程式碼（只複 app 也可）
COPY app /app/app


# 預設環境變數（compose 會覆寫）
ENV HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf

# 預設用 uvicorn 啟動；compose 可覆蓋
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

