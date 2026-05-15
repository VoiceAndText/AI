import torch
import librosa
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

def get_preprocessors():
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    return audio_processor, tokenizer

def preprocess_input(audio_path, text, audio_processor, tokenizer, device):
    # 오디오 처리 (16kHz 로드 및 5초 고정)
    audio, _ = librosa.load(audio_path, sr=16000)
    audio_values = audio_processor(
        audio, return_tensors="pt", sampling_rate=16000,
        padding='max_length', max_length=80000, truncation=True
    ).input_values.to(device)
    
    # 텍스트 처리
    text_inputs = tokenizer(
        text, return_tensors="pt", padding='max_length', max_length=128, truncation=True
    ).to(device)
    
    return audio_values, text_inputs['input_ids'], text_inputs['attention_mask']
