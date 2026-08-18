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


def check_export(project: Project, run: str | None = None,
                 formats: list[str] | None = None, images: int = 8,
                 iou: float = 0.6, conf: float = 0.25,
                 strict_conf: float = 0.5) -> int:
    """Сравнить боксы исходных весов и экспортированной модели.

    Экспорт ломается тихо: другая нормализация входа, другой порядок выходов,
    потерянный NMS — метрики при этом никто не пересчитывает, и мусор уезжает
    в приложение. Здесь одни и те же кадры прогоняются через .pt и через
    экспорт, а боксы сравниваются по IoU.

    Важно, в каком окружении запускать: CoreML читается только тем
    coremltools, что в обучающем .venv (в .venv-export версия старше и
    падает по assert внутри MPSGraph), а TFLite — наоборот, только в
    .venv-export. Поэтому `make export-check` запускает форматы порознь.
    """
    from ultralytics import YOLO

    from .analysis import IMAGE_EXTENSIONS, _iou

    source_weights = project.paths.weights(run)
    if not source_weights.is_file():
        raise SystemExit(f"Нет весов {source_weights}")

    selected = formats or list(project.get("export.formats", ["coreml", "tflite"]))
    model_name = str(project.get("export.model_name", project.name))
    targets: list[Path] = []
    for fmt in selected:
        subdirectory, extension = FORMAT_LAYOUT[fmt]
        candidate = project.paths.exports / subdirectory / f"{model_name}{extension}"
        if candidate.exists():
            targets.append(candidate)
        else:
            console.warn(f"{fmt}: нет {candidate} — сначала mlkit export {project.name}")
    if not targets:
        console.warn("Нечего проверять: нужных экспортов нет")
        return 0

    pool = project.paths.samples if project.paths.samples.is_dir() else None
    if pool is None or not any(p.suffix.lower() in IMAGE_EXTENSIONS for p in pool.iterdir()):
        pool = project.paths.dataset / "val" / "images"
    picked = sorted(p for p in pool.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)[:images]
    if not picked:
        raise SystemExit(f"Нет картинок для проверки в {pool}")

    size = int(project.get("model.imgsz", 640))
    console.step(f"Проверка экспорта на {len(picked)} кадрах")
    console.kv("эталон", source_weights)
    console.kv("порог conf / IoU", f"{conf} / {iou}")
    console.kv("считаем ошибкой при conf ≥", strict_conf)

    def boxes_of(model, image: Path) -> list[tuple]:
        # По одному кадру: CoreML-бэкенд ultralytics не умеет батчи
        result = model.predict(str(image), imgsz=size, conf=conf, verbose=False)[0]
        return [(*box.xyxyn[0].tolist(), float(box.conf)) for box in result.boxes]

    reference = YOLO(str(source_weights))
    expected = [boxes_of(reference, image) for image in picked]

    failed = False
    for target in targets:
        console.rule(target.name)
        try:
            exported = YOLO(str(target), task="detect")
            produced = [boxes_of(exported, image) for image in picked]
        except Exception as error:
            console.err(f"модель не запустилась: {type(error).__name__}: {error}")
            failed = True
            continue

        checked = 0
        worst = 1.0
        critical = 0
        borderline = 0

        def closest(box: tuple, pool: list[tuple]) -> float:
            return max((_iou(box[:4], other[:4]) for other in pool), default=0.0)

        for image, want, got in zip(picked, expected, produced):
            for box in want:                       # что эталон нашёл, а экспорт — нет
                best = closest(box, got)
                checked += 1
                if best >= iou:
                    worst = min(worst, best)
                    continue
                if box[4] >= strict_conf:
                    console.err(f"{image.name}: бокс conf={box[4]:.2f} потерян "
                                f"(лучший IoU {best:.2f})")
                    critical += 1
                else:
                    borderline += 1
            for box in got:                        # что экспорт придумал сверх эталона
                if closest(box, want) >= iou:
                    continue
                if box[4] >= strict_conf:
                    console.err(f"{image.name}: лишний бокс conf={box[4]:.2f}")
                    critical += 1
                else:
                    borderline += 1

        if critical:
            console.err(f"критичных расхождений: {critical} — в приложение не годится")
            failed = True
        else:
            console.ok(f"совпадает с эталоном: сверено боксов {checked}, "
                       f"худший IoU {worst:.3f}")
        if borderline:
            console.warn(
                f"пограничных расхождений (conf < {strict_conf}): {borderline} — "
                "разное подавление дубликатов в NMS, на качестве почти не сказывается"
            )

    if failed:
        console.info("  частые причины: NMS (export.coreml_nms), int8-квантизация, "
                     "другой imgsz на стороне приложения")
        return 1
    return 0


def export_status(project: Project) -> list[str]:
    root = project.paths.exports
    if not root.is_dir():
        return []
    return [str(path.relative_to(root)) for path in sorted(root.glob("*/*"))]
