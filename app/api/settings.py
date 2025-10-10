import os
import torch


# ---------- 基本設定 ----------
BASE_CKPT = os.getenv("BASE_CKPT", "D:\SD_LORA_Agent\models\Stable-diffusion\sd_xl_base_1.0.safetensors")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DISABLE_SAFETY = os.getenv("DISABLE_SAFETY", "1") == "1"  # 預設關閉 safety checker
A = 1.0