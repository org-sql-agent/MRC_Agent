FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 系統相依（視需求微調）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先裝依賴
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 再複製程式碼（只複 app 也可）
COPY app /app/app

# 預設用 uvicorn 啟動；compose 可覆蓋
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

