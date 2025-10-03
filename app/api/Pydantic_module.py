from pydantic import BaseModel, Field, validator
from typing import List, Optional



# ---------- Pydantic 模型 ----------
class LoRAItem(BaseModel):
    path: str = Field(..., description="D:\\SD_LORA_Agent\\models\\Lora\\MoldyV1.safetensors")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="LoRA 影響力")
    name: Optional[str] = Field(None, description="adapter 名稱；預設用檔名")

class GenerateReq(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "lowres, blurry, artifacts"
    width: int = Field(1024, description="建議 512/640/768/896/1024；必須為 8 的倍數")
    height: int = Field(1024, description="建議 512/640/768/896/1024；必須為 8 的倍數")
    steps: int = Field(80, ge=20, le=200)
    guidance_scale: float = Field(24.0, ge=3.0, le=30.0)
    image_guidance_scale: Optional[float] = Field(None, ge=0.0, le=30.0, description="A1111 image_cfg_scale（SDXL 常見 1.0~2.0）")
    seed: int = Field(-1, description="-1 表示隨機")
    vae: Optional[str] = Field(None, description="與 A1111 相同的 VAE 權重路徑或識別字")

    loras: List[LoRAItem] = Field(default_factory=list, description="可一次套多個 LoRA")

class GenerateResp(BaseModel):
    image_base64: str
    seed: int
    width: int
    height: int
    steps: int
    guidance_scale: float
    applied_loras: List[str]


class Img2ImgReq(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "lowres, blurry, artifacts"
    width: Optional[int] = Field(None, description="若要覆寫由來源圖推導的尺寸，需為 8 的倍數")
    height: Optional[int] = Field(None, description="若要覆寫由來源圖推導的尺寸，需為 8 的倍數")
    image_base64: str  # 來源圖（可含 data:image/...;base64, 前綴）
    strength: float = Field(0.35, ge=0.0, le=1.0, description="去噪幅度；0=幾乎不改, 1=幾乎重繪")
    steps: int = Field(80, ge=20, le=200)
    guidance_scale: float = Field(24.0, ge=3.0, le=30.0)
    image_guidance_scale: Optional[float] = Field(None, ge=0.0, le=30.0, description="A1111 image_cfg_scale（SDXL 常見 1.0~2.0）")
    seed: int = Field(-1, description="-1 表示隨機")
    vae: Optional[str] = Field(None, description="與 A1111 相同的 VAE 權重")
    loras: List[LoRAItem] = Field(default_factory=list)