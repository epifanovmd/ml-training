"""Поиск пар «изображение + разметка» в источниках YOLO-формата."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Roboflow при экспорте размножает кадр аугментациями, добавляя к имени
# «.rf.<хэш>». Такие кадры — один и тот же снимок, и в разные сплиты они
# попадать не должны.
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
GROUP_MODES = ("stem", "roboflow", "parent")


@dataclass(frozen=True)
class Pair:
    image: Path
    label: Path | None
    prefix: str          # префикс источника — разводит одинаковые имена файлов
    group: str = ""      # ключ группы: вся группа уходит в один сплит

    @property
    def key(self) -> str:
        return f"{self.prefix}_{self.image.stem}"


def source_prefix(source: Path) -> str:
    """Короткий стабильный префикс источника.

    Считается от имени каталога, а не от полного пути: перенос источника
    в другое место не перемешивает сплит train/val.
    """
    digest = hashlib.sha1(source.name.encode("utf-8")).hexdigest()[:6]
    return f"{source.name[:24]}-{digest}"


def label_for(image: Path) -> Path | None:
    """Разметка рядом с изображением либо в соседнем labels/."""
    candidates = [
        image.parent.parent / "labels" / f"{image.stem}.txt",
        image.parent / "labels" / f"{image.stem}.txt",
        image.with_suffix(".txt"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def group_key(image: Path, source: Path, mode: str) -> str:
    """Ключ группы кадра.

    Кадры с одинаковым ключом гарантированно попадают в один сплит. Это
    единственная защита от утечки: аугментированные копии одного снимка
    (Roboflow) или несколько ракурсов одного объекта в train и val сразу
    завышают метрики, и по ним нельзя ни выбирать веса, ни сравнивать модели.
    """
    if mode.startswith("regex:"):
        pattern = re.compile(mode.split(":", 1)[1])
        match = pattern.search(image.stem)
        if not match:
            return image.stem
        return match.group(1) if match.groups() else match.group(0)
    if mode == "parent":
        # Кадры лежат в <что-то>/images/, поэтому группируем по каталогу выше
        directory = image.parent.parent if image.parent.name == "images" else image.parent
        try:
            return directory.relative_to(source).as_posix() or "."
        except ValueError:
            return directory.name
    if mode == "roboflow":
        return ROBOFLOW_SUFFIX.sub("", image.stem)
    if mode == "stem":
        return image.stem
    raise SystemExit(
        f"Неизвестный dataset.group_by «{mode}». "
        f"Доступные: {', '.join(GROUP_MODES)} или regex:<шаблон>"
    )


def find_pairs(source: Path, group_by: str = "stem") -> list[Pair]:
    prefix = source_prefix(source)
    pairs: list[Pair] = []
    for image in sorted(source.rglob("*")):
        if image.suffix.lower() not in IMAGE_EXTENSIONS or not image.is_file():
            continue
        label = label_for(image)
        if label is not None:
            pairs.append(Pair(image=image, label=label, prefix=prefix,
                              group=f"{prefix}/{group_key(image, source, group_by)}"))
    return pairs


def looks_augmented(pairs: list[Pair]) -> int:
    """Сколько кадров похожи на аугментированные копии (Roboflow-экспорт)."""
    return sum(1 for pair in pairs if ROBOFLOW_SUFFIX.search(pair.image.stem))


def source_class_names(source: Path) -> dict[int, str]:
    """Имена классов источника из его data.yaml, если он есть.

    Roboflow и подобные экспорты кладут рядом data.yaml с `names`. По именам
    сопоставлять классы надёжнее, чем по номерам: у разных источников
    нумерация своя, и молчаливая путаница классов — самая дорогая ошибка
    при слиянии датасетов.
    """
    candidates = [source / "data.yaml", source / "data.yml"]
    candidates += sorted(source.glob("*/data.y*ml"))[:3]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            document = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        names = document.get("names")
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(value) for index, value in enumerate(names)}
    return {}


def resolve_class_map(raw: dict | None, source: Path, source_names: dict[int, str],
                      target_classes: list[str]) -> dict[int, int] | None:
    """Карта «класс источника -> класс проекта» из dataset.class_map.

    Ключи и значения задаются номерами или именами; для карт по именам
    нужен data.yaml источника (имена берутся оттуда). Поддерживается общая
    карта и карта на источник: {default: {...}, roboflow-a: {...}}.
    """
    if not raw:
        return None
    if all(isinstance(value, dict) for value in raw.values()):
        raw = raw.get(source.name, raw.get("default"))
        if not raw:
            return None

    by_name = {name.lower(): index for index, name in source_names.items()}
    targets = {name.lower(): index for index, name in enumerate(target_classes)}
    mapping: dict[int, int] = {}
    for key, value in raw.items():
        if isinstance(key, int) or str(key).lstrip("-").isdigit():
            source_index = int(key)
        elif str(key).lower() in by_name:
            source_index = by_name[str(key).lower()]
        else:
            known = ", ".join(source_names.values()) or "имён нет (нет data.yaml)"
            raise SystemExit(
                f"class_map: в источнике {source.name} нет класса «{key}». "
                f"Известные: {known}"
            )

        if isinstance(value, int) or str(value).lstrip("-").isdigit():
            target_index = int(value)
        elif str(value).lower() in targets:
            target_index = targets[str(value).lower()]
        else:
            raise SystemExit(
                f"class_map: «{value}» нет в classes проекта "
                f"({', '.join(target_classes)})"
            )
        mapping[source_index] = target_index
    return mapping


def parse_label(path: Path, keep_classes: set[int] | None = None,
                mapping: dict[int, int] | None = None, collapse: bool = True,
                num_classes: int = 1) -> tuple[list[str], int, Counter, set[int]]:
    """Строки разметки, приведённые к классам проекта.

    Возвращает (строки, отброшено, счётчик классов, классы вне диапазона).
    `collapse` — свалить все классы источника в 0 (одноклассовая задача).
    """
    lines: list[str] = []
    dropped = 0
    counts: Counter = Counter()
    out_of_range: set[int] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            dropped += 1 if parts else 0
            continue
        try:
            source_class = int(float(parts[0]))
            coordinates = [float(value) for value in parts[1:5]]
        except ValueError:
            dropped += 1
            continue
        if keep_classes is not None and source_class not in keep_classes:
            continue
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            dropped += 1
            continue
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            dropped += 1
            continue

        if collapse:
            target = 0
        elif mapping is not None:
            if source_class not in mapping:
                continue            # класса нет в карте — считаем его лишним
            target = mapping[source_class]
        else:
            target = source_class
        if not 0 <= target < num_classes:
            out_of_range.add(source_class)
            continue

        counts[target] += 1
        lines.append(f"{target} " + " ".join(f"{v:.6f}" for v in coordinates))
    return lines, dropped, counts, out_of_range
