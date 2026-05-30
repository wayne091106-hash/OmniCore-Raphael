import os
import sys
import queue
import torch
import numpy as np
import sounddevice as sd

# ==========================================
# ⚙️ 系統核心參數設定
# ==========================================
SAMPLE_RATE = 16000      # Silero VAD 官方指定的標準採樣率 (16kHz)
CHUNK_SIZE = 512         # 每次讀取的音訊訊框大小 (32ms，平衡延遲與精準度的最佳值)
VAD_THRESHOLD = 0.55     # 人聲判定機率閾值 (0.0 ~ 1.0)，數值越高判定越嚴格
GAIN_FACTOR = 2.5        # 軟體音量放大倍數 (補償 NVIDIA Broadcast 降噪後聲音偏小的問題)

# 建立執行緒安全的音訊緩衝佇列
audio_queue = queue.Queue()

# ==========================================
# 🔌 【你的專屬對接端口】
#     未來不論你要把音訊接到哪裡，都只要修改這個類別即可！
# ==========================================
class VoiceOutputPort:
    def __init__(self):
        # 這裡可以初始化你的 API 客戶端、WebSocket 連線或檔案寫入器
        self.output_destination = "Terminal Console"
        print(f"📦 [Port] 語音輸出端口已初始化。當前目的地: {self.output_destination}")

    def send_audio(self, audio_chunk, speech_probability):
        """
        當 VAD 判定【有人聲】時，會即時觸發這個函式。
        
        參數:
          audio_chunk: 一維的 numpy float32 陣列，長度為 512，代表這 32ms 的純淨人聲音訊。
          speech_probability: 模型判定的目前人聲機率 (0.55 ~ 1.00)
        """
        # 計算此訊框的物理振幅 (音量)
        rms = np.sqrt(np.mean(audio_chunk**2))
        volume_bar = "█" * int(rms * 50)
        
        # 模擬輸出動作：這裡你可以直接換成發送 API、寫入暫存檔等
        print(f"🟢 [語音傳輸中] 機率: {speech_probability:.2f} | 音量: {rms:.4f} {volume_bar:<20}", end="\r")
        
        # 例：如果是要接 WebSocket，你可以這樣寫：
        # websocket.send(audio_chunk.tobytes())

    def send_silence(self):
        """
        當 VAD 判定【無聲/雜音】時，會即時觸發這個函式。
        這代表閘門關閉，不向外部輸出任何音訊訊號。
        """
        # 在終端機上印出無聲狀態，你可以把這個 print 註解掉，讓背景保持安靜
        print("🔇 [閘門關閉] 檢測為無聲或背景環境音...", end="\r")


# ==========================================
# 🧠 載入與設定 Silero VAD 模型
# ==========================================
def initialize_silero_vad():
    """
    載入 Silero VAD。
    此模型由 PyTorch 官方代管，首次執行會自動下載至您的使用者快取路徑：
    C:\\Users\\<您的使用者名稱>\\.cache\\torch\\hub\\snakers4_silero-vad_master
    """
    print("\n[VAD 引擎] 正在載入全球最精準的 Silero VAD 模型...")
    
    try:
        # 透過 Torch Hub 載入模型結構與預訓練權重
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,  # 若本地已有快取則不重新下載，達到瞬間啟動
            trust_repo=True      # 信任開源倉庫
        )
        print("[VAD 引擎] 模型權重載入成功！")
        return model
        
    except Exception as e:
        print(f"\n❌ 模型載入失敗！請確認網路連線是否正常。")
        print(f"詳細錯誤訊息: {e}")
        sys.exit(1)


# ==========================================
# 🎙️ 音訊輸入裝置 (NVIDIA Broadcast) 綁定
# ==========================================
def get_nvidia_microphone_id():
    """
    掃描系統所有的音訊輸入裝置，並精準綁定 NVIDIA Broadcast 虛擬麥克風。
    如果使用者沒開 Broadcast，會自動提示並退回系統預設裝置。
    """
    devices = sd.query_devices()
    device_id = None
    
    for i, dev in enumerate(devices):
        # 尋找名稱包含 NVIDIA 且具備輸入軌道的裝置
        if 'NVIDIA Broadcast' in dev['name'] and dev['max_input_channels'] > 0:
            device_id = i
            print(f"🎯 [硬體綁定] 成功對接 RTX 硬體降噪裝置：[ID: {i}] {dev['name']}")
            break
            
    if device_id is None:
        print("\n⚠️  [硬體提示] 未偵測到 NVIDIA Broadcast 虛擬裝置。")
        print("   建議先開啟 NVIDIA Broadcast 軟體，否則系統將採用【預設麥克風】。")
        # 選擇預設輸入裝置
        device_id = sd.default.device[0]
        try:
            default_dev_name = devices[device_id]['name']
            print(f"🔄 [自動備援] 已切換為系統預設輸入裝置：{default_dev_name}")
        except Exception:
            print("❌ 無法取得任何麥克風輸入裝置。")
            sys.exit(1)
            
    return device_id


# ==========================================
# 🔄 音訊非同步串流回呼
# ==========================================
def audio_callback(indata, frames, time, status):
    """
    這個函式會由系統底層的音訊驅動程式(Sounddevice)以極高的優先權持續呼叫。
    我們只做一件事：迅速把資料丟入佇列，避免阻塞音訊緩衝區，造成爆音。
    """
    if status:
        print(f"\n⚠️  [音訊驅動回報]: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())


# ==========================================
# 🚀 系統主程式進入點
# ==========================================
def run_main_loop():
    # 1. 初始化對接端口與 VAD 模型
    port = VoiceOutputPort()
    vad_model = initialize_silero_vad()
    
    # 2. 獲取音訊輸入裝置 ID
    input_device_id = get_nvidia_microphone_id()
    
    print("\n========================================================")
    print(" 🚀 Silero VAD 智能閘門系統已開始監聽！")
    print("    - 請對著麥克風說話測試...")
    print("    - 終端機將即時顯示判定狀態...")
    print("    - 欲結束請按: Ctrl + C")
    print("========================================================\n")
    
    try:
        # 3. 建立並開啟 PyTorch 音訊串流 (InputStream)
        with sd.InputStream(samplerate=SAMPLE_RATE, 
                            blocksize=CHUNK_SIZE,
                            device=input_device_id,
                            dtype='float32',
                            channels=1, 
                            callback=audio_callback):
            
            while True:
                # 從緩衝佇列中取得最新 32ms 的音訊片段 (若無資料則阻塞等待)
                raw_chunk = audio_queue.get()
                
                # 執行你在前一步驟設定的軟體增益，放大音量
                # 並限制最大/最小振幅在 -1.0 到 1.0 之間，防止數位失真爆音
                processed_chunk = np.clip(raw_chunk * GAIN_FACTOR, -1.0, 1.0)
                
                # 將音訊轉換為 PyTorch 的 FloatTensor 格式 [batch_size, sequence_length]
                audio_tensor = torch.from_numpy(processed_chunk).view(1, -1)
                
                # 呼叫 Silero VAD 進行即時推論，回傳此訊框是人聲的機率
                speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
                
                # 4. 根據 VAD 機率，決定是否開啟閘門並發送至端口
                if speech_prob >= VAD_THRESHOLD:
                    # 將二維 Tensor 攤平回一維 numpy 陣列，方便後續 API 處理
                    output_data = processed_chunk.flatten()
                    port.send_audio(output_data, speech_prob)
                else:
                    port.send_silence()
                    
    except KeyboardInterrupt:
        print("\n\n停止監聽。已安全關閉 VAD 閘門與音訊流。")
    except Exception as e:
        print(f"\n❌ 運作時發生未預期錯誤: {e}")

if __name__ == "__main__":
    run_main_loop()