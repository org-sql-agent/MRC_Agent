import os, io, base64, json, requests
from PIL import Image
from openai import OpenAI

from openai_env import SD_API

# 你允許 LLM 使用的 LoRA 名稱與路徑（只需在這裡維護）
LORA_REGISTRY = {
    "Moldy_SDXL_V0": r"/models/Lora/Moldy_SDXL_V0.safetensors",
    # "AnimeV2": r"D:\...\AnimeV2.safetensors",
}




# ========= 小工具 =========
def img_to_data_uri(file, mime: str = "image/png") -> str:
    """將使用者上傳的圖轉為 data URL（送 LLM & SD 用）"""
    if isinstance(file, bytes):
        raw = file
    else:
        raw = file.read()
    # 轉 PNG 保險（也可直接用原檔）
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        mime = "image/png"
    except Exception:
        pass
    return f"data:{mime};base64," + base64.b64encode(raw).decode("utf-8")

def loras_from_names(items):
    """將 LLM 回傳的 loras 名稱映射成 SD 後端需要的 {path, weight, name}"""
    out = []
    for it in items or []:
        name = it["name"]
        weight = float(it.get("weight", 0.8))
        if name not in LORA_REGISTRY:
            continue
        out.append({"path": LORA_REGISTRY[name], "weight": weight, "name": name})
    return out

def call_sd_txt2img(args: dict) -> dict:
    payload = {
        "prompt": args["prompt"],
        "negative_prompt": args.get("negative_prompt", "lowres, blurry, artifacts"),
        "width": args.get("width", 1024),
        "height": args.get("height", 1024),
        "steps": args.get("steps", 80),
        "guidance_scale": args.get("guidance_scale", 24.5),
        "seed": args.get("seed", -1),
        "loras": loras_from_names(args.get("loras", [])),
    }
    r = requests.post(f"{SD_API}/generate", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()

def call_sd_img2img(args: dict, uploaded_image_b64: str | None) -> dict:
    if not uploaded_image_b64:
        raise ValueError("你選了 img2img，但沒有上傳參考圖。")
    payload = {
        "prompt": args["prompt"],
        "negative_prompt": args.get("negative_prompt", "lowres, blurry, artifacts"),
        "image_base64": uploaded_image_b64,   # 由前端補入
        "strength": args.get("strength", 0.35),
        "steps": args.get("steps", 80),
        "guidance_scale": args.get("guidance_scale", 24.5),
        "seed": args.get("seed", -1),
        "loras": loras_from_names(args.get("loras", [])),
    }
    r = requests.post(f"{SD_API}/img2img", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()