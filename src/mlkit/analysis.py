"""Разбор качества и скорости обученной модели.

Две вещи, которых не видно по одной цифре mAP: на каких кадрах модель
ошибается (и каких данных не хватает) и сколько миллисекунд она стоит.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import console
from .config import Project
from .paths import resolve

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GT_COLOR = "#06D6A0"      # разметка
PRED_COLOR = "#FF3D71"    # предсказание


@dataclass
class Frame:
    """Итог сравнения предсказаний и разметки на одном кадре."""

    image: Path
    missed: list[tuple]        # не найдено (FN): x1,y1,x2,y2,cls
    spurious: list[tuple]      # лишнее (FP): x1,y1,x2,y2,conf,cls
    matched: int
    confused: int = 0          # бокс на месте, но класс другой

    @property
    def errors(self) -> int:
        return len(self.missed) + len(self.spurious)


def _boxes_from_label(path: Path) -> list[tuple]:
    """YOLO-строки -> (x1, y1, x2, y2, класс) в нормированных координатах."""
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, width, height = (float(value) for value in parts[1:5])
        boxes.append((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2,
                      int(float(parts[0]))))
    return boxes


def _iou(first: tuple, second: tuple) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if overlap <= 0:
        return 0.0
    area = ((first[2] - first[0]) * (first[3] - first[1])
            + (second[2] - second[0]) * (second[3] - second[1]) - overlap)
    return overlap / area if area > 0 else 0.0


def _match(truth: list[tuple], predicted: list[tuple], threshold: float,
           multiclass: bool) -> Frame:
    """Жадное сопоставление по IoU: что нашли, что пропустили, что придумали.

    При нескольких классах бокс на нужном месте, но с другим классом —
    отдельная категория: это не «пропуск», а путаница, и лечится она иначе.
    """
    used: set[int] = set()
    matched = confused = 0
    missed = []
    for box in truth:
        best, best_iou = -1, 0.0
        for index, candidate in enumerate(predicted):
            if index in used:
                continue
            value = _iou(box[:4], candidate[:4])
            if value > best_iou:
                best, best_iou = index, value
        if best >= 0 and best_iou >= threshold:
            used.add(best)
            same_class = (not multiclass) or box[4] == predicted[best][5]
            matched += 1 if same_class else 0
            confused += 0 if same_class else 1
            if not same_class:
                missed.append(box)
        else:
            missed.append(box)
    spurious = [box for index, box in enumerate(predicted) if index not in used]
    return Frame(image=Path("."), missed=missed, spurious=spurious, matched=matched,
                 confused=confused)


def _draw(frame: Frame, target: Path, classes: list[str] | None = None) -> None:
    """Кадр с разметкой (зелёная) и предсказаниями (розовые)."""
    from PIL import Image, ImageDraw

    image = Image.open(frame.image).convert("RGB")
    width, height = image.size
    canvas = ImageDraw.Draw(image)
    stroke = max(2, round(min(width, height) * 0.004))

    def label_of(index: int) -> str:
        return classes[index] if classes and 0 <= index < len(classes) else str(index)

    for box in frame.missed:
        canvas.rectangle((box[0] * width, box[1] * height,
                          box[2] * width, box[3] * height), outline=GT_COLOR, width=stroke)
        if classes and len(classes) > 1:
            canvas.text((box[0] * width + 4, box[1] * height + 4),
                        label_of(box[4]), fill=GT_COLOR)
    for box in frame.spurious:
        canvas.rectangle((box[0] * width, box[1] * height,
                          box[2] * width, box[3] * height), outline=PRED_COLOR, width=stroke)
        if classes and len(classes) > 1 and len(box) > 5:
            canvas.text((box[0] * width + 4, box[3] * height - 14),
                        f"{label_of(box[5])} {box[4]:.2f}", fill=PRED_COLOR)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, quality=90)


def errors(project: Project, run: str | None = None, split: str = "val",
           conf: float | None = None, iou: float = 0.5, limit: int = 20,
           device: str | None = None) -> int:
    """Найти кадры, где модель ошибается, и сохранить их с боксами."""
    from ultralytics import YOLO

    weights = project.paths.weights(run)
    if not weights.is_file():
        raise SystemExit(f"Нет весов {weights} — сначала обучите модель")
    images_dir = project.paths.dataset / split / "images"
    labels_dir = project.paths.dataset / split / "labels"
    if not images_dir.is_dir():
        raise SystemExit(f"Нет сплита «{split}» в {project.paths.dataset}")

    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    threshold = float(conf if conf is not None else project.get("predict.conf", 0.25))
    console.step(f"Разбор ошибок на сплите {split}: {len(images)} кадров")
    console.kv("веса", weights)
    console.kv("порог conf / IoU", f"{threshold} / {iou}")

    model = YOLO(str(weights))
    kwargs = {"device": device} if device else {}
    multiclass = len(project.classes) > 1
    frames: list[Frame] = []
    totals = {"missed": 0, "spurious": 0, "matched": 0, "clean": 0, "confused": 0}
    progress = console.Progress("кадров", total=len(images))
    for image, result in zip(images, model.predict(
        [str(path) for path in images], imgsz=int(project.get("model.imgsz", 640)),
        conf=threshold, stream=True, verbose=False, **kwargs,
    )):
        predicted = [(*box.xyxyn[0].tolist(), float(box.conf), int(box.cls))
                     for box in result.boxes]
        truth = _boxes_from_label(labels_dir / f"{image.stem}.txt")
        frame = _match(truth, predicted, iou, multiclass)
        frame = Frame(image=image, missed=frame.missed, spurious=frame.spurious,
                      matched=frame.matched, confused=frame.confused)
        totals["missed"] += len(frame.missed)
        totals["spurious"] += len(frame.spurious)
        totals["matched"] += frame.matched
        totals["confused"] += frame.confused
        totals["clean"] += 1 if frame.errors == 0 else 0
        if frame.errors:
            frames.append(frame)
        progress.advance(1)
    progress.close()

    console.rule("итог")
    console.kv("кадров без ошибок", f"{totals['clean']} из {len(images)}")
    console.kv("найдено верно", totals["matched"])
    console.kv("пропущено (FN)", totals["missed"])
    console.kv("лишних срабатываний (FP)", totals["spurious"])
    if multiclass:
        console.kv("перепутан класс", totals["confused"])

    if not frames:
        console.ok("Ошибок нет — сохранять нечего")
        return 0

    frames.sort(key=lambda frame: (-frame.errors, frame.image.name))
    target = project.paths.previews / f"errors-{split}"
    if target.exists():
        import shutil

        shutil.rmtree(target)
    for position, frame in enumerate(frames[:limit], 1):
        name = (f"{position:02d}_fn{len(frame.missed)}_fp{len(frame.spurious)}"
                + (f"_cls{frame.confused}" if frame.confused else "")
                + f"_{frame.image.name}")
        _draw(frame, target / name, project.classes)
    console.ok(f"Худшие {min(limit, len(frames))} кадров: {target}")
    console.info(f"  зелёный — пропущенная разметка, розовый — лишнее срабатывание")
    return 0


def benchmark(project: Project, run: str | None = None, weights: str | None = None,
              imgsz: int | None = None, device: str | None = None,
              repeats: int = 50, image: str | None = None,
              only: list[str] | None = None) -> int:
    """Замерить время одного кадра. Работает и для .pt, и для экспортов.

    `only` ограничивает список расширений: мобильные форматы читаются
    в разных окружениях (CoreML — в .venv, TFLite — в .venv-export),
    поэтому make-цели замеряют их порознь.
    """
    from ultralytics import YOLO

    candidates = [resolve(weights)] if weights else _bench_targets(project, run)
    if only:
        wanted = {f".{value.lstrip('.')}" for value in only}
        candidates = [path for path in candidates if path.suffix in wanted]
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        raise SystemExit("Нечего замерять: нет ни весов прогона, ни экспортов")

    source = resolve(image) if image else _sample_image(project)
    size = int(imgsz or project.get("model.imgsz", 640))
    console.step(f"Замер скорости на {source.name}, {repeats} прогонов")
    console.kv("imgsz", size)
    console.kv("устройство", device or "по умолчанию")

    for path in candidates:
        try:
            model = YOLO(str(path), task="detect")
            kwargs = {"device": device} if device else {}
            for _ in range(3):                       # прогрев
                model.predict(str(source), imgsz=size, verbose=False, **kwargs)
            started = time.perf_counter()
            for _ in range(repeats):
                model.predict(str(source), imgsz=size, verbose=False, **kwargs)
            elapsed = (time.perf_counter() - started) / repeats * 1000
            size_mb = _tree_size(path) / 1e6
            console.kv(path.name, f"{elapsed:6.1f} мс/кадр   {1000 / elapsed:5.1f} FPS   "
                                  f"{size_mb:.1f} МБ")
        except Exception as error:
            hint = ("— мобильные форматы замеряются в окружении экспорта: "
                    f"make bench-export P={project.name}"
                    if path.suffix in (".tflite", ".mlpackage") else
                    f"— {type(error).__name__}: {error}")
            console.warn(f"{path.name}: не запустилось {hint}")
    return 0


def _bench_targets(project: Project, run: str | None) -> list[Path]:
    """Веса прогона плюс всё, что уже экспортировано."""
    targets = [project.paths.weights(run)]
    exports = project.paths.exports
    if exports.is_dir():
        for pattern in ("ios/*.mlpackage", "android/*.tflite", "onnx/*.onnx"):
            targets.extend(sorted(exports.glob(pattern)))
    return targets


def _sample_image(project: Project) -> Path:
    for directory in (project.paths.samples, project.paths.dataset / "val" / "images"):
        if directory.is_dir():
            images = sorted(p for p in directory.iterdir()
                            if p.suffix.lower() in IMAGE_EXTENSIONS)
            if images:
                return images[0]
    raise SystemExit(
        f"Нет картинки для замера: положите файл в {project.paths.samples} "
        "или укажите --image"
    )


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
