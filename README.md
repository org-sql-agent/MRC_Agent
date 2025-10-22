# MRC Agent


---
## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/org-sql-agent/MRC_Agent.git
cd MRC_Agent
```

### 2. build the venv in MRC_Agent
```bash
python3.10 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```


### 2. Create a `.env` File
Inside the project root, create a `.env` file with the following content:
```env
OPENAI_API_KEY=your_api_key_here
MODELS_DIR=your model folder
```


### 3. Create model folder

```plaintext
MRC_Agent/
└── models/
    ├── Stable-diffusion/
    │   └── sd_xl_base_1.0.safetensors
    └── Lora/
        └── Moldy_SDXL_V0.safetensors
```

Base model download：https://drive.google.com/file/d/1JRowubDodm_H6rSr9pmIyTdOlOyBozsR/view?usp=drive_link

Lora model download: https://drive.google.com/file/d/1HOCUoQK_VXKSeZFehhfM5-IqS9TOHx_-/view?usp=drive_link

### 4. Run 

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
cd app/webui
streamlit run app.py --server.port 8501
```

After the service starts, open your browser and visit:
```
http://localhost:8501
```

You should now see the **MRC Agent** interface.


FastAPI backend :
```
http://127.0.0.1:8000
```

