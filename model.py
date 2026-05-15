import torch
import torch.nn as nn
from transformers import WavLMModel, RobertaModel

class CrossAttentionFusionModel(nn.Module):
    def __init__(self, input_dim=768, num_classes=2):
        super(CrossAttentionFusionModel, self).__init__()
        self.audio_to_text_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        self.text_to_audio_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        self.ln_audio = nn.LayerNorm(input_dim)
        self.ln_text = nn.LayerNorm(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio, text):
        a = audio.unsqueeze(1) if audio.dim() == 2 else audio
        t = text.unsqueeze(1) if text.dim() == 2 else text
        attn_t, _ = self.text_to_audio_attn(t, a, a)
        attn_a, _ = self.audio_to_text_attn(a, t, t)
        combined_a = self.ln_audio(audio + attn_a.squeeze(1))
        combined_t = self.ln_text(text + attn_t.squeeze(1))
        merged = torch.cat((combined_a, combined_t), dim=1)
        return self.classifier(merged)

class EndToEndStressModel(nn.Module):
    def __init__(self):
        super(EndToEndStressModel, self).__init__()
        self.audio_encoder = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.text_encoder = RobertaModel.from_pretrained("klue/roberta-base")
        self.fusion_layer = CrossAttentionFusionModel()

    def forward(self, audio, ids, mask):
        a_out = self.audio_encoder(audio).last_hidden_state.mean(dim=1)
        t_out = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
        return self.fusion_layer(a_out, t_out)
