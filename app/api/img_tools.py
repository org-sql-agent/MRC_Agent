import os
import base64
from typing import List, Optional
from app.api.Pydantic_module import LoRAItem
from app.api.pipeline import pipe, pipe_i2i, LOADED_ADAPTERS, GPU_LOCK
from fastapi import HTTPException
from PIL import Image
from io import BytesIO
# ---------- 工具函式 ----------


def _adapter_name_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def _ensure_loras_loaded(loras: List[LoRAItem]) -> List[str]:
    adapter_names, weights = [], []
    
    for l in loras:
        name = l.name or _adapter_name_from_path(l.path)
        if name not in LOADED_ADAPTERS:
            p = l.path
            try:
                # 1) 路徑存在嗎？
                if not os.path.exists(p):
                    raise FileNotFoundError(f"LoRA 檔案/資料夾不存在: {p}")

                # 2) 單檔 vs 資料夾：優先判斷「單一 .safetensors 檔」
                if os.path.isfile(p):
                    dir_path = os.path.dirname(p) or "."
                    fname = os.path.basename(p)

                    # 嘗試 A：新式 API：資料夾 + weight_name=檔名
                    try:
                        pipe.load_lora_weights(dir_path, weight_name=fname, adapter_name=name)
                    except Exception as e_a:
                        # 嘗試 B：直接給單檔路徑（部分 diffusers 版本可用）
                        try:
                            pipe.load_lora_weights(p, adapter_name=name)
                        except Exception as e_b:
                            raise RuntimeError(
                                f"LoRA 單檔載入失敗。\n"
                                f"- 方法A(dir+weight_name={fname})錯誤：{e_a}\n"
                                f"- 方法B(直接單檔)錯誤：{e_b}"
                            )
                else:
                    # 3) 資料夾（或 repo）模式：讓 diffusers 自行尋找預設檔名
                    pipe.load_lora_weights(p, adapter_name=name)

                LOADED_ADAPTERS.add(name)

            except Exception as e:
                # 轉成 400，讓客戶端能看到可讀錯誤，而不是只有 500
                raise HTTPException(status_code=400, detail=f"LoRA 載入失敗：{p} -> {e}")  # 參考伺服器端結構:contentReference[oaicite:4]{index=4}

        adapter_names.append(name)
        weights.append(l.weight)

    if adapter_names:
        # 同步到 txt2img 與 img2img 兩條管線
        pipe.set_adapters(adapter_names, adapter_weights=weights)
        try:
            pipe_i2i.set_adapters(adapter_names, adapter_weights=weights)
        except Exception:
            pass
    else:
        try:
            pipe.disable_lora()
            pipe_i2i.disable_lora()
        except Exception:
            pass

    return adapter_names



def _to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _from_b64(data_uri: str) -> Image.Image:
    # 允許帶或不帶 data:image/png;base64, 前綴
    b64 = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    return img


def autosize_within_cap(orig_w: int, orig_h: int, cap: int = 1024) -> tuple[int, int]:
    """
    若原圖的長邊超過 cap，就等比縮到長邊 == cap；否則維持原尺寸。
    回傳值會再做 8 的倍數對齊，避免 SDXL 管線報錯。
    """
    max_side = max(orig_w, orig_h)
    if max_side <= cap:
        new_w, new_h = orig_w, orig_h
    else:
        scale = cap / float(max_side)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

    # 對齊 8 的倍數（避免很多 VAE / U-Net 報尺寸錯）
    new_w = max(8, new_w - (new_w % 8))
    new_h = max(8, new_h - (new_h % 8))
    return new_w, new_h

def _resize_to_multiple_of_8(img: Image.Image) -> Image.Image:
    w, h = img.size
    nw , nh = autosize_within_cap(w, h)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), resample=Image.LANCZOS)
    return img
