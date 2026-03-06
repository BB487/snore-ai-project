import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
import os
import io
import pandas as pd
from pydub import AudioSegment

# --- 字體設定 ---
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 參數設定 ---
MODEL_PATH = "crnn_expert_model_v15_merged.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 44100
THRESHOLD = 0.80
SILENCE_THRESHOLD = 0.003
MIN_SNORE_VOLUME = 0.015
N_MELS = 128; N_FFT = 2048; HOP_LENGTH = 512

# --- AI 模型架構 ---
class CRNNExpertModel(nn.Module):
    def __init__(self, num_classes):
        super(CRNNExpertModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.rnn_input_dim = 128 * 8
        self.rnn = nn.GRU(input_size=self.rnn_input_dim, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention_layer = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        batch_size, time_steps, C, freq_dim = x.size()
        x = x.reshape(batch_size, time_steps, -1)
        x, _ = self.rnn(x)
        attn_weights = F.softmax(self.attention_layer(x), dim=1)
        x = torch.sum(x * attn_weights, dim=1)
        return self.classifier(x)

@st.cache_resource
def load_model():
    model = CRNNExpertModel(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

def preprocess_audio_denoised(audio_data, sr=SAMPLE_RATE):
    try: audio_data = nr.reduce_noise(y=audio_data, sr=sr, prop_decrease=0.8, n_fft=1024, stationary=True)
    except: pass
    if np.max(np.abs(audio_data)) > 0: audio_data = audio_data / np.max(np.abs(audio_data))
    melspec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    target_len = 345
    if melspec_db.shape[1] < target_len:
        melspec_db = np.pad(melspec_db, ((0, 0), (0, target_len - melspec_db.shape[1])), mode='constant', constant_values=-80.0)
    else: melspec_db = melspec_db[:, :target_len]
    return torch.tensor(melspec_db[np.newaxis, np.newaxis, :, :]).float()

# --- 網頁介面設計 ---
st.set_page_config(page_title="AI 睡眠鼾聲檢測站", layout="wide", page_icon="🌙")

# 【升級點 3：側邊欄說明與科普】
with st.sidebar:
    st.title("🩺 關於本系統")
    st.info("本系統利用深度學習 (CRNN) 自動分析睡眠錄音中的鼾聲，提供快速的居家前置篩檢。")
    st.markdown("### 📝 使用步驟")
    st.markdown("1. **準備音檔**：請準備 .wav 格式的睡眠錄音。")
    st.markdown("2. **上傳檔案**：拖曳至右側上傳區。")
    st.markdown("3. **AI 分析**：點擊開始分析，等待報告產出。")
    st.markdown("4. **下載報告**：可將圖表下載帶給醫師參考。")
    st.markdown("---")
    st.markdown("### 💡 醫學小知識：什麼是 SI？")
    st.markdown("SI (Snore Index) 代表**每小時打呼次數**。數值越高，代表睡眠呼吸中止症 (OSA) 的風險可能越高，建議尋求專業醫師協助。")

st.title("🌙 AI 居家睡眠鼾聲篩檢系統")
st.markdown("請上傳您的睡眠錄音檔 (.wav)，AI 將為您分析整晚的打呼狀況。")

uploaded_file = st.file_uploader("上傳錄音檔", type=['wav'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("🚀 開始 AI 深度分析", use_container_width=True):
        with st.spinner("AI 正在聆聽並分析您的睡眠狀況，請稍候..."):
            temp_path = "temp_upload.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            model = load_model()
            snore_events, full_wave_downsampled = [], []
            DOWNSAMPLE_FACTOR = 100
            
            with sf.SoundFile(temp_path) as f:
                sr, channels = f.samplerate, f.channels
                total_frames = len(f)
                window_size, step_size = int(sr * 4.0), int(sr * 1.0)
                
                progress_bar = st.progress(0)
                current_frame = 0
                
                while current_frame + window_size <= total_frames:
                    f.seek(current_frame)
                    audio_segment = f.read(window_size)
                    if channels > 1: audio_segment = np.mean(audio_segment, axis=1)
                    if len(audio_segment) < N_FFT:
                        current_frame += step_size
                        continue
                        
                    rms = np.sqrt(np.mean(audio_segment**2))
                    if rms >= SILENCE_THRESHOLD:
                        input_tensor = preprocess_audio_denoised(audio_segment, sr).to(DEVICE)
                        with torch.no_grad():
                            output = model(input_tensor)
                            prob = F.softmax(output, dim=1)[0][0].item()
                        if prob > THRESHOLD and rms > MIN_SNORE_VOLUME:
                            snore_events.append((current_frame / sr, (current_frame + step_size) / sr))
                    
                    full_wave_downsampled.append(audio_segment[:step_size:DOWNSAMPLE_FACTOR])
                    current_frame += step_size
                    progress_bar.progress(min(current_frame / total_frames, 1.0))
            
            # 產生圖表
            full_wave = np.concatenate(full_wave_downsampled)
            time_axis = np.arange(len(full_wave)) * (DOWNSAMPLE_FACTOR / sr)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 3]})
            plt.subplots_adjust(hspace=0.4)
            ax1.set_title("Snore Events Timeline", fontsize=16, fontweight='bold')
            for start, end in snore_events: ax1.axvspan(start, end, color='red', alpha=0.5)
            ax1.set_xlim(0, time_axis[-1]); ax1.set_yticks([]); ax1.set_xlabel("Time (seconds)")
            
            ax2.set_title("Acoustic Waveform & AI Prediction", fontsize=16, fontweight='bold')
            ax2.plot(time_axis, full_wave, color='lightgray', linewidth=0.5)
            for start, end in snore_events:
                s_idx, e_idx = int(start * sr / DOWNSAMPLE_FACTOR), int(end * sr / DOWNSAMPLE_FACTOR)
                ax2.plot(time_axis[s_idx:e_idx], full_wave[s_idx:e_idx], color='red', linewidth=0.8)
            ax2.set_xlabel("Time (seconds)"); ax2.set_ylabel("Amplitude")
            
            duration_hrs = max(time_axis[-1], 1) / 3600
            si = len(snore_events) / duration_hrs
            severity = "正常 (Normal)" if si < 5 else "輕度打呼 (Mild)" if si < 15 else "中度打呼 (Moderate)" if si < 30 else "重度打呼，建議就醫 (Severe)"
            
            st.success("✅ 分析完成！")
            st.pyplot(fig)
            
            # 【升級點 2：圖表下載按鈕】
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                label="📥 下載醫療分析圖表",
                data=buf,
                file_name="Snore_Medical_Report.png",
                mime="image/png"
            )
            
            st.markdown("---")
            st.markdown("### 📊 睡眠聲學分析報告")
            col1, col2, col3 = st.columns(3)
            col1.metric("監測時長", f"{duration_hrs:.2f} 小時")
            col2.metric("打呼總次數", f"{len(snore_events)} 次")
            col3.metric("SI 鼾聲指數", f"{si:.1f} 次/小時")
            
            if si >= 30: st.error(f"**臨床評估：{severity}**")
            elif si >= 15: st.warning(f"**臨床評估：{severity}**")
            else: st.info(f"**臨床評估：{severity}**")
            
            st.caption("備註：本系統提供之 SI (Snore Index) 為聲學初篩指標，非最終醫療診斷。若需確診睡眠呼吸中止症 (OSA)，請至醫院進行 AHI 檢測。")

            # 【升級點 1：精確打呼時間紀錄表】
            st.markdown("---")
            st.markdown("### ⏱️ 打呼事件詳細紀錄")
            if len(snore_events) > 0:
                event_data = []
                for idx, (start, end) in enumerate(snore_events):
                    m_start, s_start = divmod(int(start), 60)
                    duration = end - start
                    event_data.append({
                        "事件編號": f"第 {idx+1} 次",
                        "發生時間 (分:秒)": f"{m_start:02d}:{s_start:02d}",
                        "絕對秒數": f"{start:.2f} s",
                        "持續長度": f"{duration:.2f} s"
                    })
                df_events = pd.DataFrame(event_data)
                st.dataframe(df_events, use_container_width=True)
            else:

                st.info("太棒了！這段錄音中沒有偵測到任何打呼聲。")
