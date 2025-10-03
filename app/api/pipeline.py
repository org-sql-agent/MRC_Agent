from app.api.settings import BASE_CKPT, DEVICE, DTYPE, DISABLE_SAFETY

from diffusers import StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
import asyncio


# ---------- 啟動 Pipeline ----------
pipe_kwargs = {"torch_dtype": DTYPE, "use_safetensors": True}
if DISABLE_SAFETY:
    pipe_kwargs.update({"safety_checker": None, "requires_safety_checker": False})

try:
    pipe = StableDiffusionXLPipeline.from_single_file(BASE_CKPT, **pipe_kwargs)
except Exception as e:
    raise RuntimeError(f"載入 base 模型失敗: {e}")

# 取樣器設定：DPM++ 2M Karras 對應
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True
)

# 移到裝置
pipe = pipe.to(DEVICE)

# 省顯存與加速（可按需關閉）

def _tune_pipe(p):
    try:
        p.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    p.enable_attention_slicing()
    p.enable_vae_slicing()
    return p
_tune_pipe(pipe)

# 建立共用元件的 img2img 管線（與 txt2img 共存）
pipe_i2i = StableDiffusionXLImg2ImgPipeline(**pipe.components)
pipe_i2i.scheduler = pipe.scheduler
pipe_i2i = pipe_i2i.to(DEVICE)
_tune_pipe(pipe_i2i)


# 載入/管理 LoRA 的簡易快取
LOADED_ADAPTERS = set()
GPU_LOCK = asyncio.Lock()