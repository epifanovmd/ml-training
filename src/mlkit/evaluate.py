"""Проверка модели «глазами»: разметка своих картинок и сводка в консоли."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import console
from .config import Project
from .paths import resolve

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def predict(project: Project, input_dir: str | Path | None = None,
            output_dir: str | Path | None = None, weights: str | None = None,
            run: str | None = None, conf: float | None = None,
            device: str | None = None, show: bool = False) -> int:
    from ultralytics import YOLO

    checkpoint = resolve(weights) if weights else project.paths.weights(run)
    if not checkpoint.is_file():
        raise SystemExit(
            f"Нет весов {checkpoint} — сначала обучите модель: mlkit train {project.name}"
        )

    source = resolve(input_dir) if input_dir else project.paths.samples
    if not source.is_dir():
        source.mkdir(parents=True, exist_ok=True)
        raise SystemExit(f"Каталог {source} создан — положите картинки и запустите снова")

    images = sorted(p for p in source.rglob("*")
                    if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    if not images:
        raise SystemExit(f"В {source} нет картинок ({', '.join(sorted(IMAGE_EXTENSIONS))})")

    target = resolve(output_dir) if output_dir else project.paths.previews / "predict"
    target.mkdir(parents=True, exist_ok=True)

    threshold = float(conf if conf is not None else project.get("predict.conf", 0.25))
    console.step(f"Проверка {checkpoint.name} на {len(images)} изображениях")
    console.kv("вход", source)
    console.kv("порог conf", threshold)

    model = YOLO(str(checkpoint))
    kwargs = {"device": device or project.get("predict.device")} \
        if (device or project.get("predict.device")) else {}
    detected = 0
    for image, result in zip(images, model.predict(
        [str(path) for path in images],
        imgsz=int(project.get("model.imgsz", 640)),
        conf=threshold, stream=True, verbose=False, **kwargs,
    )):
        result.save(str(target / image.name))
        boxes = result.boxes
        detected += 1 if len(boxes) else 0
        console.info(f"{image.name}: объектов {len(boxes)}")
        for box in boxes:
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            console.info(f"    {result.names[int(box.cls)]} {float(box.conf):.2f} "
                         f"[{x1}, {y1}, {x2}, {y2}]")

    console.ok(f"С объектами: {detected} из {len(images)}")
    console.info(f"  результат: {target}")
    if show and sys.platform == "darwin":
        subprocess.run(["open", str(target)], check=False)
    return 0
