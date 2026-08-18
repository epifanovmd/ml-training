"""Раскладка групп по train/val/test и её сохранение между сборками.

Сплит считается по группам (см. discover.group_key). Отдельный модуль нужен
из-за «липкости»: при пополнении датасета уже известные группы обязаны
остаться в своих сплитах, иначе метрики соседних прогонов несравнимы, а
тёплый старт с прошлых весов превращается в утечку — модель уже видела
кадры, которые теперь оказались в val.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")
STATE_VERSION = 1


def _order(groups: list[str]) -> list[str]:
    """Детерминированный порядок групп: одинаковый на любой машине."""
    return sorted(set(groups), key=lambda key: hashlib.sha1(key.encode("utf-8")).hexdigest())


def assign_splits(groups: list[str], val_ratio: float,
                  test_ratio: float = 0.0) -> dict[str, str]:
    """Разложить группы по сплитам: детерминированно и с соблюдением долей.

    Группы упорядочиваются по хэшу имени, доли отрезаются квотами. Чистое
    пороговое хэширование на малом числе групп даёт перекос вплоть до
    пустого val; квоты этого не допускают, оставаясь воспроизводимыми.
    """
    ordered = _order(groups)
    total = len(ordered)
    if total == 0:
        return {}

    def quota(ratio: float) -> int:
        if ratio <= 0 or total < 2:
            return 0
        return max(1, min(total - 1, round(total * ratio)))

    count_val = quota(val_ratio)
    count_test = quota(test_ratio)
    while count_val + count_test >= total and count_test > 0:
        count_test -= 1
    while count_val + count_test >= total and count_val > 0:
        count_val -= 1

    assignment = {}
    for position, key in enumerate(ordered):
        if position < count_val:
            assignment[key] = "val"
        elif position < count_val + count_test:
            assignment[key] = "test"
        else:
            assignment[key] = "train"
    return assignment


def extend_assignment(previous: dict[str, str], groups: list[str], val_ratio: float,
                      test_ratio: float = 0.0) -> tuple[dict[str, str], int]:
    """Сохранить сплиты известных групп и разложить только новые.

    Новая группа отправляется в тот сплит, которому сильнее всего не хватает
    до целевой доли, — так пропорции подтягиваются по мере пополнения, а
    старые кадры никуда не переезжают. Возвращает (раскладка, сколько новых).
    """
    ordered = _order(groups)
    known = {key: previous[key] for key in ordered if key in previous}
    fresh = [key for key in ordered if key not in previous]
    if not known:
        return assign_splits(groups, val_ratio, test_ratio), len(fresh)

    assignment = dict(known)
    counts = {split: sum(1 for value in assignment.values() if value == split)
              for split in SPLITS}
    targets = {"val": val_ratio, "test": test_ratio,
               "train": max(0.0, 1.0 - val_ratio - test_ratio)}

    for key in fresh:
        total = sum(counts.values()) + 1
        # выбираем сплит с наибольшим дефицитом относительно целевой доли
        deficit = {split: targets[split] - counts[split] / total
                   for split in SPLITS if targets[split] > 0}
        assignment[key] = max(deficit, key=deficit.get) if deficit else "train"
        counts[assignment[key]] += 1
    return assignment, len(fresh)


def signature(assignment: dict[str, str]) -> str:
    """Короткий отпечаток раскладки — по нему видно, что сплит менялся."""
    payload = ";".join(f"{key}={value}" for key, value in sorted(assignment.items()))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(path: Path, assignment: dict[str, str], group_by: str,
               val_ratio: float, test_ratio: float) -> None:
    """Хранится рядом с рабочим каталогом, а не в датасете: датасет стирается."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "version": STATE_VERSION,
        "group_by": group_by,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "signature": signature(assignment),
        "assignment": assignment,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
