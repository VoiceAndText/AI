# 실행 명령어: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
import os
import requests
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisper
import librosa

# 코드 임포트
from model import EndToEndStressModel
from preprocess import get_preprocessors

app = FastAPI(title="V&T Real Inference API")

# --- 전역 변수 세팅 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}

# 백엔드에서 받을 JSON 규격 (Presigned URL)
class AnalyzeRequest(BaseModel):
    file_url: str

@app.on_event("startup")
async def load_models():
    """서버 시작 시 무거운 AI 모델들을 메모리에 한 번만 로드합니다."""
    print(f"🚀 AI 모델 로딩 시작... (Device: {device})")
    
    # 1. Whisper 모델 로드
    models["whisper"] = whisper.load_model("base").to(device)
    
    # 2. 전처리기 로드
    audio_processor, tokenizer = get_preprocessors()
    models["audio_processor"] = audio_processor
    models["tokenizer"] = tokenizer
    
    # 3. 융합 모델 로드
    vnt_model = EndToEndStressModel().to(device)
    
    vnt_model.load_state_dict(torch.load("best_model.pth", map_location=device))
    
    vnt_model.eval() # 추론 모드로 전환 (Dropout 등 비활성화)
    models["vnt_model"] = vnt_model
    
    print("✅ 모든 AI 모델 로딩 완료!")

@app.post("/api/v1/analyze")
async def analyze_audio(req: AnalyzeRequest):
    """
    S3 Presigned URL을 받아 다운로드 후, STT 분리 및 V&T 모델 추론을 수행합니다.
    """
    temp_audio_path = "temp_downloaded.wav"
    
    try:
        # 1. S3 Presigned URL에서 오디오 파일 다운로드
        print("\n[Step 1] S3에서 오디오 다운로드 중...")
        response = requests.get(req.file_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="S3 URL에서 파일을 다운로드할 수 없습니다.")
        
        with open(temp_audio_path, "wb") as f:
            f.write(response.content)
            
        # 2. Whisper로 문장 단위 분리 및 인메모리 슬라이싱
        print("[Step 2] Whisper STT 문장 분리 진행 중...")
        stt_result = models["whisper"].transcribe(temp_audio_path, language="ko", condition_on_previous_text=False)
        waveform, sr = librosa.load(temp_audio_path, sr=16000)
        
        final_report = []
        dissonance_sum = 0.0
        valid_chunk_count = 0
        
        print("[Step 3] 멀티모달 V&T 추론 시작...")
        for idx, seg in enumerate(stt_result["segments"]):
            start_time, end_time = seg["start"], seg["end"]
            text = seg["text"].strip()
            
            # 노이즈/환각 필터링
            if (end_time - start_time) < 1.0 or len(text.split()) < 2:
                continue
                
            # Numpy 슬라이싱
            start_sample, end_sample = int(start_time * sr), int(end_time * sr)
            audio_chunk = waveform[start_sample:end_sample]
            
            # 원본 코드는 파일 경로를 받지만, 이미 잘라놓은 audio_chunk(numpy)를 넣도록 수정
            audio_values = models["audio_processor"](
                audio_chunk, return_tensors="pt", sampling_rate=16000,
                padding='max_length', max_length=80000, truncation=True
            ).input_values.to(device)
            
            text_inputs = models["tokenizer"](
                text, return_tensors="pt", padding='max_length', max_length=128, truncation=True
            ).to(device)
            # --------------------------------------------------------
            
            # 3. 모델 추론
            with torch.no_grad():
                logits = models["vnt_model"](
                    audio_values, 
                    text_inputs['input_ids'], 
                    text_inputs['attention_mask']
                )
                # Logits(num_classes=2)를 Softmax로 확률값(0~1)으로 변환
                probs = F.softmax(logits, dim=1).squeeze(0) 
                
                # 가설: Class 0 = 정상(일치), Class 1 = 불일치(Conflict)
                conflict_prob = probs[1].item() * 100 
                
            is_conflict = bool(conflict_prob > 60.0) # 60% 이상이면 불일치로 판정 (Threshold)
            
            final_report.append({
                "time_range": f"{start_time:.1f}s - {end_time:.1f}s",
                "stt_chunk": text,
                "text_emotion": "분석중", # 현재 모델이 이진 분류라 임시 텍스트
                "audio_emotion": "분석중",
                "dissonance_score": round(conflict_prob, 2),
                "is_conflict": is_conflict
            })
            
            dissonance_sum += conflict_prob
            valid_chunk_count += 1
            print(f"  -> [{start_time:.1f}s] {text} | 불일치율: {conflict_prob:.1f}%")

        # 4. 종합 결과 계산
        overall_dissonance = round(dissonance_sum / valid_chunk_count, 2) if valid_chunk_count > 0 else 0.0
        primary_emotion = "불일치 감지 (스트레스)" if overall_dissonance > 50 else "안정 (일치)"

        return {
            "status": "success",
            "message": "AI 멀티모달 감정 불일치 분석이 완료되었습니다.",
            "data": {
                "overall_analysis": {
                    "primary_emotion": primary_emotion,
                    "dissonance_index": overall_dissonance
                },
                "time_series_analysis": final_report
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 분석 서버 에러: {str(e)}")
    
    finally:
        # 다운로드했던 임시 파일 삭제 (용량 관리)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)