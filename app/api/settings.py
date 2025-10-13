import os
import torch


# ---------- 基本設定 ----------
BASE_CKPT = os.getenv("BASE_CKPT", "/models/Stable-diffusion/sd_xl_base_1.0.safetensors")
DEVICE = "mps" if torch.mps.is_available() else "cpu"
DTYPE = torch.float32 if torch.mps.is_available() else torch.float16
DISABLE_SAFETY = os.getenv("DISABLE_SAFETY", "1") == "1"  # 預設關閉 safety checker
