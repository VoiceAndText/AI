# 실행 명령어: uvicorn mock_api_server:app --host 0.0.0.0 --port 8000 --reload
from fastapi import FastAPI, UploadFile, File
import time

app = FastAPI(title="V&T Mock Inference API (Dissonance Focus)")

@app.post("/api/v1/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    백엔드(Spring) 연동 테스트를 위한 가짜(Mock) API 엔드포인트
    """
    # 1. 실제 분석이 일어나는 것처럼 지연 시간(Latency) 시뮬레이션
    time.sleep(2.0)
    
    # 2. 백엔드 개발자에게 전달할 약속된 JSON 포맷 반환
    return {
        "status": "success",
        "message": "AI 멀티모달 감정 불일치 분석이 완료되었습니다.",
        "data": {
            "file_name": file.filename,
            "overall_analysis": {
                "primary_emotion": "부정", 
                "dissonance_index": 78.5  # 전체 대화의 평균 감정 불일치 지수 (0~100)
            },
            "time_series_analysis": [
                {
                    "time_range": "0.0s - 3.5s",
                    "stt_chunk": "아 진짜 너무 행복하다.",
                    "text_emotion": "긍정",   # 텍스트(RoBERTa)가 판단한 감정
                    "audio_emotion": "부정",  # 음성(WavLM)이 판단한 감정
                    "dissonance_score": 92.0, # 어텐션 스코어 기반 불일치 정도
                    "is_conflict": True       # 불일치 여부 (Threshold 기준)
                },
                {
                    "time_range": "3.5s - 7.0s",
                    "stt_chunk": "오늘 일도 다 안 끝났는데.",
                    "text_emotion": "부정",
                    "audio_emotion": "부정",
                    "dissonance_score": 15.0,
                    "is_conflict": False
                }
            ]
        }
    }
