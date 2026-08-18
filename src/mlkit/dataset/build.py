"""Сборка train/val из размеченных источников."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from .. import console
from ..config import Project, dataset_sources
from ..paths import resolve
from .discover import Pair, find_pairs, parse_label, split_of


def _sources(project: Project, override: list[str] | None) -> list[Path]:
    if override:
        found = [resolve(value) for value in override]
        missing = [path for path in found if not path.exists()]
        if missing:
            raise SystemExit("Нет источников: " + ", ".join(str(p) for p in missing))
        return found
    sources = dataset_sources(project)
    if not sources:
        raise SystemExit(
            f"Нет источников данных для «{project.name}».\n"
            f"  Положите размеченный датасет в datasets/{project.name}/<имя>/\n"
            f"  (пары images/*.jpg + labels/*.txt в YOLO-формате)"
        )
    return sources


def build_dataset(project: Project, sources: list[str] | None = None,
                  verify: bool = False, val_ratio: float | None = None) -> int:
    """Собрать datasets -> workspace/<проект>/dataset. Каталог пересоздаётся."""
    source_dirs = _sources(project, sources)
    ratio = float(val_ratio if val_ratio is not None
                  else project.get("dataset.val_ratio", 0.15))
    keep = project.get("dataset.keep_classes")
    keep_classes = {int(value) for value in keep} if keep else None
    classes = project.classes

    console.step(f"Сборка датасета «{project.name}»")
    pairs: list[Pair] = []
    for source in source_dirs:
        found = find_pairs(source)
        console.kv(str(source), f"{len(found)} пар")
        if not found:
            console.warn(f"{source}: пар изображение+разметка не найдено")
        pairs.extend(found)

    if not pairs:
        console.err("Ни одной пары изображение+разметка")
        return 1

    seen: dict[str, Pair] = {}
    for pair in pairs:
        seen.setdefault(pair.key, pair)
    if len(seen) != len(pairs):
        console.warn(f"Одинаковых имён между источниками: {len(pairs) - len(seen)} "
                     "(взял первое вхождение)")
    pairs = list(seen.values())

    console.kv("всего пар", len(pairs))
    console.kv("val_ratio", ratio)
    console.kv("классы", ", ".join(classes))
    if verify:
        console.ok("Проверка завершена, ничего не записано")
        return 0

    root = project.paths.dataset
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)

    hard_link = not bool(project.get("dataset.copy", True))
    counters = {"train": 0, "val": 0, "empty": 0, "boxes": 0, "dropped": 0}
    progress = console.Progress("файлов", total=len(pairs))
    for pair in pairs:
        lines, dropped = parse_label(pair.label, keep_classes) if pair.label else ([], 0)
        counters["dropped"] += dropped
        counters["boxes"] += len(lines)
        if not lines:
            counters["empty"] += 1
        split = split_of(pair.key, ratio)
        target = root / split / "images" / f"{pair.key}{pair.image.suffix.lower()}"
        _place(pair.image, target, hard_link)
        (root / split / "labels" / f"{pair.key}.txt").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )
        counters[split] += 1
        progress.advance(1)
    progress.close()

    data_yaml = {
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "names": {index: name for index, name in enumerate(classes)},
    }
    with project.paths.data_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data_yaml, fh, allow_unicode=True, sort_keys=False)

    console.kv("train", counters["train"])
    console.kv("val", counters["val"])
    console.kv("боксов", counters["boxes"])
    console.kv("негативных (без боксов)", counters["empty"])
    if counters["dropped"]:
        console.warn(f"отброшено битых строк разметки: {counters['dropped']}")
    console.ok(f"Готово: {project.paths.data_yaml}")
    console.info(f"  дальше: mlkit train {project.name}")
    return 0


def _place(source: Path, target: Path, hard_link: bool) -> None:
    if hard_link:
        try:
            os.link(source.resolve(), target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def dataset_status(project: Project) -> dict[str, int]:
    root = project.paths.dataset
    if not project.paths.data_yaml.is_file():
        return {}
    return {
        split: len(list((root / split / "images").glob("*")))
        for split in ("train", "val")
        if (root / split / "images").is_dir()
    }


def dataset_stats(project: Project) -> int:
    """Подробности по собранному датасету: боксы, размеры, негативные кадры."""
    root = project.paths.dataset
    if not project.paths.data_yaml.is_file():
        raise SystemExit(f"Датасет не собран — mlkit dataset build {project.name}")

    console.step(f"Датасет «{project.name}»: {root}")
    total_boxes = 0
    for split in ("train", "val"):
        labels = sorted((root / split / "labels").glob("*.txt"))
        boxes = 0
        empty = 0
        areas: list[float] = []
        for label in labels:
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line]
            boxes += len(lines)
            empty += 0 if lines else 1
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    areas.append(float(parts[3]) * float(parts[4]))
        total_boxes += boxes
        console.rule(split)
        console.kv("изображений", len(labels))
        console.kv("боксов", boxes)
        console.kv("негативных", empty)
        if areas:
            areas.sort()
            median = areas[len(areas) // 2]
            console.kv("медианная площадь бокса", f"{median * 100:.2f}% кадра")
            console.kv("самый мелкий", f"{areas[0] * 100:.3f}% кадра")
    console.info("")
    console.kv("итого боксов", total_boxes)
    return 0
