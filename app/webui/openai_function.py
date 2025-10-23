
from tools import LORA_REGISTRY

# ========= OpenAI 工具（functions）Schema =========

SYSTEM_PROMPT = f"""
你是黴菌影像生成協調器。你只能在必要時呼叫工具（txt2img 或 img2img），
並且參數要簡潔、有效且可重現（適度使用 seed）。
規則：
1) 若「使用者有上傳圖片」 → 選 img2img。
2) 若無上傳圖片或需求是從零生成 → 選 txt2img。
3) steps 建議 80到100。
4) guidance_scale 大一點建議至少 24.0；尺寸以 512~1024 的 8 倍數；必要時才超過。
5) 呼叫img2img時，strength小一點，盡量保持輸入圖的原型，建議0.35。 
4) LoRA 僅能使用這些名稱：{", ".join(LORA_REGISTRY.keys())}；務必填 weight，預設weight為1.0。
5) 你的回應只能是工具呼叫（不要聊天），除非明確無法執行。
6) 在生成"prompt"與"negative_prompt"時，你只能使用英文生成。
7) 在生成"prompt"時，一定要提到有關"moldy"相關詞彙，有助於喚醒lora。
8) 在生成"prompt"時，最後要加上"<Moldy_SDXL_V0>"，有助於喚醒lora。
9) 在生成"prompt"時，不要超過70 tokens。
10) 你只能使用 JSON 格式回應工具呼叫（符合 OpenAI function calling 規範）。
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "txt2img",
            "description": "用 Stable Diffusion + LoRA 從文字生成圖片（無輸入圖時選這個）",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "width": {"type": "integer", "minimum": 64, "maximum": 2048},
                    "height": {"type": "integer", "minimum": 64, "maximum": 2048},
                    "steps": {"type": "integer", "minimum": 20, "maximum": 200},
                    "guidance_scale": {"type": "number", "minimum": 3.0, "maximum": 30.0},
                    "seed": {"type": "integer"},
                    "loras": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "enum": list(LORA_REGISTRY.keys())},
                                "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "required": ["name", "weight"],
                            "additionalProperties": False
                        },
                        "default": []
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "img2img",
            "description": "用 Stable Diffusion + LoRA 對上傳圖片做風格/內容轉換（有輸入圖時選這個）",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "steps": {"type": "integer", "minimum": 20, "maximum": 200},
                    "guidance_scale": {"type": "number", "minimum": 3.0, "maximum": 30.0},
                    "seed": {"type": "integer"},
                    "loras": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "enum": list(LORA_REGISTRY.keys())},
                                "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "required": ["name", "weight"],
                            "additionalProperties": False
                        },
                        "default": []
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False
            },
        },
    },
]