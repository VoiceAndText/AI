# 1. 파이썬 3.9 슬림 버전 이미지를 베이스로 사용
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. Whisper 구동에 필수인 ffmpeg 및 빌드 도구 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. 요구사항 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 및 모델 가중치 파일 복사
# (.pth 파일도 같은 폴더에 있어야 도커 이미지 안에 포함됩니다)
COPY . .

# 6. FastAPI 포트 개방
EXPOSE 8000

# 7. 서버 실행 명령어
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]