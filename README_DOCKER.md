Dockerized MRC_Agent
====================

Quick steps to build and run the MRC_Agent project in Docker. This repo contains heavy model files under `models/` so we recommend mounting that directory at runtime rather than baking it into the image.

Build image (CPU-only, no GPU):

```bash
docker build -t mrc-agent:latest .
```

Run (mount local `models/`):

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" mrc-agent:latest
```

Run with GPU (NVIDIA Docker, ensures CUDA available):

```bash
docker build -t mrc-agent:latest .
docker run --gpus all --rm -p 8000:8000 -v "$(pwd)/models:/app/models" mrc-agent:latest
```

Using docker-compose (GPU machines):

```bash
docker-compose up --build
```

Notes & recommendations
- The base image already includes matching CUDA + PyTorch; requirements.txt contains a torch spec which we filter out to avoid reinstalling a different wheel. If you need different CUDA/toolkit versions, change the base image to the appropriate `pytorch/pytorch:<tag>`.
- Keep `models/` as an external volume to avoid large image sizes. The Dockerfile defines `/app/models` as a VOLUME for this reason.
- If you prefer Poetry-managed environments, you can adapt the Dockerfile to `poetry install --no-dev`. I elected to use `pip install -r requirements.txt` for simplicity in containers.
- If the project uses GPU and large VRAM models, consider limiting workers to 1 and pinning device usage in the code.
