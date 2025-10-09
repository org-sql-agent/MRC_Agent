
from fastapi import FastAPI

from app.api.settings import DEVICE , DTYPE
from app.api.Pydantic_module import GenerateReq, GenerateResp, Img2ImgReq
from app.api.pipeline import pipe, pipe_i2i, LOADED_ADAPTERS, GPU_LOCK
from app.api.img_tools import _ensure_loras_loaded, _from_b64, _to_b64, _resize_to_multiple_of_8, autosize_within_cap
import torch

app = FastAPI(title="SDXL + LoRA Agent (Diffusers)")



# ---------- 路由 ----------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "dtype": str(DTYPE), "loaded_adapters": list(LOADED_ADAPTERS)}

@app.post("/generate", response_model=GenerateResp)    #呼叫生成模型
async def generate(req: GenerateReq):
    w,h = autosize_within_cap(req.width, req.height)
    # w = _round_to_multiple_of(req.width)
    # h = _round_to_multiple_of(req.height)

    # 產生器（seed 可重現）
    if req.seed is None or req.seed < 0:
        seed = torch.seed() % (2**32)
    else:
        seed = int(req.seed)
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    # 載入/啟用 LoRA
    adapters = _ensure_loras_loaded(req.loras)

    # 單 GPU 保守做法：序列化存取，避免 OOM / 競態
    async with GPU_LOCK:
        image = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=w,
            height=h,
            generator=g,
        ).images[0]

    return GenerateResp(
        image_base64=_to_b64(image),
        seed=seed,
        width=w,
        height=h,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        applied_loras=adapters
    )


@app.post("/img2img", response_model=GenerateResp)
async def img2img(req: Img2ImgReq):
    # 產生器（seed 可重現）
    if req.seed is None or req.seed < 0:
        seed = torch.seed() % (2**32)
    else:
        seed = int(req.seed)
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    # 載入/啟用 LoRA（同步套用到兩條管線）
    adapters = _ensure_loras_loaded(req.loras)

    # 讀入並規整尺寸（8 的倍數）
    init_img = _resize_to_multiple_of_8(_from_b64(req.image_base64))

    async with GPU_LOCK:
        out = pipe_i2i(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=init_img,
            strength=req.strength,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            generator=g,
        ).images[0]

    return GenerateResp(
        image_base64=_to_b64(out),
        seed=seed,
        width=out.width,
        height=out.height,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        applied_loras=adapters
    )