"""Стадия 4 — обучение. Все гиперпараметры приходят из project.yaml."""

from __future__ import annotations

from pathlib import Path

from . import console
from .config import Project
from .paths import WEIGHTS_DIR, resolve


def resolve_base_weights(value: str) -> str:
    """Базовые веса: локальный файл, кэш weights/ или загрузка ultralytics."""
    path = Path(str(value)).expanduser()
    if path.is_file():
        return str(path)
    if len(path.parts) == 1:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        cached = WEIGHTS_DIR / path.name
        if cached.is_file():
            return str(cached)
        try:
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(str(cached))
            if cached.is_file():
                console.ok(f"Базовые веса загружены: {cached}")
                return str(cached)
        except Exception as error:
            console.warn(f"Не удалось положить веса в {WEIGHTS_DIR}: {error}")
    return str(value)


def run_name_for(project: Project, name: str | None, base_model: str | None) -> str:
    if name:
        return name
    if base_model:
        return f"{project.name}-{Path(base_model).stem}"
    return project.name


def train(project: Project, epochs: int | None = None, device: str | None = None,
          batch: int | None = None, base_model: str | None = None,
          imgsz: int | None = None, name: str | None = None,
          resume: bool = False, extra: dict | None = None) -> int:
    from ultralytics import YOLO

    data_yaml = project.paths.data_yaml
    if not data_yaml.is_file():
        raise SystemExit(
            f"Нет {data_yaml} — сначала соберите датасет: mlkit dataset build {project.name}"
        )

    settings = dict(project.get("train", {}))
    if epochs is not None:
        settings["epochs"] = epochs
    if device is not None:
        settings["device"] = device
    if batch is not None:
        settings["batch"] = batch
    settings.update(extra or {})

    weights = resolve_base_weights(base_model or project.get("model.base", "yolo11n.pt"))
    size = int(imgsz or project.get("model.imgsz", 640))
    run = run_name_for(project, name, base_model)
    if resume:
        last = project.paths.run_dir(run) / "weights" / "last.pt"
        if not last.is_file():
            raise SystemExit(f"Нечего продолжать: нет {last}")
        weights = str(last)

    console.step(f"Обучение «{project.name}» → runs/{run}")
    console.kv("датасет", data_yaml)
    console.kv("базовые веса", weights)
    console.kv("imgsz", size)
    console.kv("эпох", settings.get("epochs"))
    console.kv("устройство", settings.get("device"))

    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml),
        imgsz=size,
        project=str(project.paths.runs),
        name=run,
        exist_ok=True,
        resume=resume,
        **settings,
        **dict(project.get("augment", {})),
    )

    # project/name задаём явно, иначе ultralytics пишет в runs/detect/ в корне
    metrics = model.val(data=str(data_yaml), imgsz=size,
                        project=str(project.paths.runs), name=f"{run}/val",
                        exist_ok=True)
    console.rule("итог")
    console.kv("mAP50", f"{metrics.box.map50:.4f}")
    console.kv("mAP50-95", f"{metrics.box.map:.4f}")
    console.kv("precision", f"{metrics.box.mp:.4f}")
    console.kv("recall", f"{metrics.box.mr:.4f}")
    console.ok(f"Лучшие веса: {Path(results.save_dir) / 'weights' / 'best.pt'}")
    hint = f"mlkit export {project.name}"
    if run != project.name:
        hint += f" --run {run}"
    console.info(f"  дальше: {hint}")
    return 0


def validate(project: Project, run: str | None = None, device: str | None = None) -> int:
    """Посчитать метрики обученной модели на val-сплите."""
    from ultralytics import YOLO

    weights = project.paths.weights(run)
    if not weights.is_file():
        raise SystemExit(f"Нет весов {weights}")
    console.step(f"Валидация {weights}")
    model = YOLO(str(weights))
    metrics = model.val(data=str(project.paths.data_yaml),
                        imgsz=int(project.get("model.imgsz", 640)),
                        project=str(project.paths.runs),
                        name=f"{run or project.name}/val", exist_ok=True,
                        **({"device": device} if device else {}))
    console.kv("mAP50", f"{metrics.box.map50:.4f}")
    console.kv("mAP50-95", f"{metrics.box.map:.4f}")
    return 0
