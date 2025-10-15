# MRC Agent


---
## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/org-sql-agent/MRC_Agent.git
cd MRC_Agent
```

### 2. Create a `.env` File
Inside the project root, create a `.env` file with the following content:
```env
OPENAI_API_KEY=your_api_key_here
MODELS_DIR=your model folder
```


### 3. Create model folder

ˋˋˋpgsql
models/
├── Stable-diffusion/
│   └── sd_xl_base_1.0.safetensors
└── Lora/
    └── Moldy_SDXL_V0.safetensors
ˋˋˋ

Base model download：https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors

Lora model download: https://drive.google.com/file/d/1HOCUoQK_VXKSeZFehhfM5-IqS9TOHx_-/view?usp=drive_link

### 4. Build and Run with Docker Compose
ˋˋˋbash
docker compose up --build
ˋˋˋ

After the service starts, open your browser and visit:
```
http://localhost:8501
```

You should now see the **MRC Agent** interface.


FastAPI backend :
ˋˋˋ
http://localhost:8000/health
ˋˋˋ

