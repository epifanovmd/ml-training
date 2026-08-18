"""Стадия 5 — конвертация обученных весов в мобильные форматы.

Выполняется в отдельном окружении .venv-export: в обучающем окружении
экспорт TFLite/CoreML ломает связку torch/torchvision (см. requirements/).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import console
from .config import Project
from .paths import PRETRAINED_DIR, resolve
from .training import resolve_base_weights

# Каталог назначения для каждого формата: iOS/Android разложены отдельно
FORMAT_LAYOUT = {
    "coreml": ("ios", ".mlpackage"),
    "tflite": ("android", ".tflite"),
    "onnx": ("onnx", ".onnx"),
    "torchscript": ("torchscript", ".torchscript"),
}


def _is_end_to_end(model) -> bool:
    """v10/v26 несут NMS внутри архитектуры — флаг nms им не передаётся."""
    return bool(getattr(getattr(model, "model", None), "end2end", False))


def export(project: Project | None, run: str | None = None, weights: str | None = None,
           formats: list[str] | None = None, out: str | None = None,
           pretrained: str | None = None, imgsz: int | None = None) -> int:
    from ultralytics import YOLO

    if pretrained:
        checkpoint = Path(resolve_base_weights(pretrained))
        size = int(imgsz or 640)
        model_name = Path(pretrained).stem
        out_root = resolve(out) if out else PRETRAINED_DIR / model_name
        coreml_nms, int8, calibration = False, False, None
        selected = formats or ["coreml", "tflite"]
    else:
        assert project is not None
        checkpoint = resolve(weights) if weights else project.paths.weights(run)
        if not checkpoint.is_file():
            raise SystemExit(
                f"Нет весов {checkpoint} — сначала обучите модель: mlkit train {project.name}"
            )
        size = int(imgsz or project.get("model.imgsz", 640))
        model_name = str(project.get("export.model_name", project.name))
        out_root = resolve(out) if out else project.paths.exports
        coreml_nms = bool(project.get("export.coreml_nms", False))
        int8 = bool(project.get("export.tflite_int8", False))
        calibration = str(project.paths.data_yaml) if int8 else None
        selected = formats or list(project.get("export.formats", ["coreml", "tflite"]))
        if int8 and not project.paths.data_yaml.is_file():
            raise SystemExit(
                "export.tflite_int8=true требует собранного датасета для калибровки"
            )

    half = bool(project.get("export.half", True)) if project else True
    console.step(f"Экспорт {checkpoint}")
    console.kv("форматы", ", ".join(selected))
    console.kv("imgsz", size)
    console.kv("имя модели", model_name)

    exported: list[Path] = []
    for fmt in selected:
        if fmt not in FORMAT_LAYOUT:
            raise SystemExit(f"Неизвестный формат «{fmt}». "
                             f"Доступные: {', '.join(FORMAT_LAYOUT)}")
        subdirectory, extension = FORMAT_LAYOUT[fmt]
        model = YOLO(str(checkpoint))          # каждый экспорт — со свежей модели
        kwargs: dict = {}
        if fmt == "coreml":
            kwargs["half"] = half
            if not _is_end_to_end(model):
                kwargs["nms"] = coreml_nms
        if fmt == "tflite":
            kwargs["int8"] = int8
            kwargs["data"] = calibration

        console.info(f"  → {fmt}…")
        produced = Path(model.export(format=fmt, imgsz=size, **kwargs))
        target_dir = out_root / subdirectory
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{model_name}{extension}"
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(produced), str(target)) if produced.is_dir() \
            else shutil.copy2(produced, target)
        exported.append(target)

    console.ok("Экспортировано (переносится в приложение вручную):")
    for path in exported:
        console.info(f"    {path}")
    return 0


def export_status(project: Project) -> list[str]:
    root = project.paths.exports
    if not root.is_dir():
        return []
    return [str(path.relative_to(root)) for path in sorted(root.glob("*/*"))]
