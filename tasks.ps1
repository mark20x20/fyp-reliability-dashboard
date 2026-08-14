param([Parameter(Position=0)][string]$Task = "help")

switch ($Task) {
  "init" {
    if (-not (Test-Path .env)) { Copy-Item .env.example .env }
    docker compose build
    docker compose run --rm app python scripts/init_db.py
    docker compose run --rm app python scripts/download_data.py
  }
  "up"      { docker compose up }
  "down"    { docker compose down }
  "shell"   { docker compose run --rm app bash }
  "gate"    { docker compose run --rm app pytest tests/test_gradcam.py -v }
  "test"    { docker compose run --rm app pytest -v }
  "train"   { docker compose run --rm app python src/train_model.py --config configs/config.yaml }
  "pipeline"{ docker compose run --rm app python run_pipeline.py --dataset imagenette }
  "smoke"   { docker compose run --rm app python run_pipeline.py --dataset imagenette --limit 100 }
  "gpu"     { docker compose run --rm app python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" }
  default   { "init | up | down | shell | gate | test | train | pipeline | smoke | gpu" }
}