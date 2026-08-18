"""Сборка train/val/test из размеченных источников."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

from .. import __version__, console
from ..config import Project, dataset_sources
from ..paths import resolve
from .discover import (Pair, find_pairs, looks_augmented, parse_label,
                        resolve_class_map, source_class_names)
from .splits import (assign_splits, extend_assignment, load_state, save_state,
                     signature)

SPLITS = ("train", "val", "test")


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
                  verify: bool = False, val_ratio: float | None = None,
                  test_ratio: float | None = None, group_by: str | None = None,
                  max_negatives: float | None = None,
                  reset_splits: bool = False) -> int:
    """Собрать datasets -> workspace/<проект>/dataset. Каталог пересоздаётся."""
    source_dirs = _sources(project, sources)
    val_share = float(val_ratio if val_ratio is not None
                      else project.get("dataset.val_ratio", 0.15))
    test_share = float(test_ratio if test_ratio is not None
                       else project.get("dataset.test_ratio", 0.0))
    grouping = str(group_by or project.get("dataset.group_by", "stem"))
    keep = project.get("dataset.keep_classes")
    keep_classes = {int(value) for value in keep} if keep else None
    classes = project.classes
    collapse = _collapse_mode(project, classes)
    raw_map = project.get("dataset.class_map")

    console.step(f"Сборка датасета «{project.name}»")
    pairs: list[Pair] = []
    per_source: list[dict] = []
    mappings: dict[str, dict[int, int] | None] = {}
    for source in source_dirs:
        found = find_pairs(source, grouping)
        groups = {pair.group for pair in found}
        names = source_class_names(source)
        mapping = resolve_class_map(raw_map, source, names, classes)
        mappings[found[0].prefix if found else str(source)] = mapping
        details = f"{len(found)} пар / {len(groups)} групп"
        if names and not collapse:
            details += f" / классы источника: {', '.join(names.values())}"
        console.kv(str(source), details)
        if not found:
            console.warn(f"{source}: пар изображение+разметка не найдено")
        per_source.append({"path": str(source), "pairs": len(found),
                           "groups": len(groups),
                           "class_map": {str(k): v for k, v in (mapping or {}).items()}})
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
    groups = {pair.group for pair in pairs}

    console.kv("всего пар", len(pairs))
    console.kv("групп (единиц сплита)", f"{len(groups)}  [group_by: {grouping}]")
    console.kv("val / test", f"{val_share} / {test_share}")
    console.kv("классы", f"{', '.join(classes)}"
                         + ("  (все классы источников сводятся в один)" if collapse else ""))
    _warn_about_leakage(pairs, groups, grouping)

    histogram, problems = _scan_classes(pairs, mappings, keep_classes, collapse, classes)
    _report_classes(histogram, classes)
    if problems:
        console.err("Классы разметки не укладываются в classes проекта:")
        for line in problems[:5]:
            console.info(f"    {line}")
        raise SystemExit(
            "Поправьте dataset.class_map (или dataset.keep_classes) — иначе часть "
            "разметки потеряется молча"
        )
    if verify:
        console.ok("Проверка завершена, ничего не записано")
        return 0

    root = project.paths.dataset
    if root.exists():
        shutil.rmtree(root)
    used_splits = ("train", "val", "test") if test_share > 0 else ("train", "val")
    for split in used_splits:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)

    hard_link = not bool(project.get("dataset.copy", True))
    counters = {split: 0 for split in SPLITS}
    counters.update({"empty": 0, "boxes": 0, "dropped": 0})
    assignment, split_note = _resolve_assignment(project, sorted(groups), val_share,
                                                 test_share, grouping, reset_splits)
    if split_note:
        console.info(f"  {split_note}")

    # Разметку читаем один раз: она же решает, кадр позитивный или фоновый
    parsed: dict[str, tuple[list[str], int]] = {}
    for pair in pairs:
        lines, dropped, _, _ = parse_label(
            pair.label, keep_classes, mappings.get(pair.prefix), collapse, len(classes)
        ) if pair.label else ([], 0, Counter(), set())
        parsed[pair.key] = (lines, dropped)

    skipped_negatives = _limit_negatives(project, pairs, parsed, assignment,
                                         max_negatives)

    progress = console.Progress("файлов", total=len(pairs) - len(skipped_negatives))
    for pair in pairs:
        if pair.key in skipped_negatives:
            continue
        lines, dropped = parsed[pair.key]
        counters["dropped"] += dropped
        counters["boxes"] += len(lines)
        if not lines:
            counters["empty"] += 1
        split = assignment[pair.group]
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
        **({"test": "test/images"} if test_share > 0 else {}),
        "names": {index: name for index, name in enumerate(classes)},
    }
    with project.paths.data_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data_yaml, fh, allow_unicode=True, sort_keys=False)

    _write_index(root, pairs, assignment, skipped_negatives)
    manifest = _write_manifest(project, root, per_source, counters, groups,
                               grouping, val_share, test_share, collapse, histogram,
                               len(skipped_negatives), signature(assignment))

    for split in used_splits:
        console.kv(split, counters[split])
    _report_negatives(counters, len(skipped_negatives))
    empty_splits = [split for split in used_splits if counters[split] == 0]
    if empty_splits:
        console.err(
            f"пустые сплиты: {', '.join(empty_splits)} — групп всего {len(groups)} "
            f"при group_by: {grouping}"
        )
        console.info("  сплит делится по группам целиком, поэтому групп должно быть "
                     "заметно больше, чем сплитов; смените dataset.group_by или "
                     "добавьте данных")
    console.kv("боксов", counters["boxes"])
    if counters["dropped"]:
        console.warn(f"отброшено битых строк разметки: {counters['dropped']}")
    console.ok(f"Готово: {project.paths.data_yaml}")
    console.info(f"  паспорт сборки: {manifest}")
    console.info(f"  дальше: mlkit train {project.name}")
    return 0


def _limit_negatives(project: Project, pairs: list[Pair],
                     parsed: dict[str, tuple[list[str], int]],
                     assignment: dict[str, str],
                     override: float | None = None) -> set[str]:
    """Ограничить долю кадров без объектов.

    Фон нужен — он снижает ложные срабатывания, — но в избытке вреден:
    модель начинает «стесняться» находить объекты, а при авторазметке
    пустая метка часто означает не «объекта нет», а «разметчик не увидел».
    Лишние негативы отбрасываются детерминированно (по хэшу имени), чтобы
    сборка оставалась воспроизводимой, и по каждому сплиту отдельно.
    """
    limit = override if override is not None else project.get("dataset.negatives.max_share")
    if limit is None:
        return set()
    limit = float(limit)
    if not 0.0 <= limit <= 1.0:
        raise SystemExit("dataset.negatives.max_share: ожидается доля от 0 до 1")

    by_split: dict[str, dict[str, list[Pair]]] = {}
    for pair in pairs:
        split = assignment[pair.group]
        kind = "positive" if parsed[pair.key][0] else "negative"
        by_split.setdefault(split, {"positive": [], "negative": []})[kind].append(pair)

    skipped: set[str] = set()
    for split, kinds in by_split.items():
        positives = len(kinds["positive"])
        negatives = kinds["negative"]
        if not negatives:
            continue
        # доля считается от итогового размера сплита: n / (p + n) <= limit
        allowed = int(positives * limit / (1 - limit)) if limit < 1.0 else len(negatives)
        if len(negatives) <= allowed:
            continue
        ordered = sorted(negatives,
                         key=lambda pair: hashlib.sha1(pair.key.encode()).hexdigest())
        skipped.update(pair.key for pair in ordered[allowed:])
    return skipped


def _report_negatives(counters: dict, skipped: int) -> None:
    """Доля фона в датасете и подсказка, если она выходит за разумные рамки."""
    total = sum(counters[split] for split in SPLITS if split in counters)
    if not total:
        return
    share = counters["empty"] / total
    console.kv("негативных (без боксов)", f"{counters['empty']} ({share * 100:.0f}%)"
                                          + (f", отброшено сверх лимита {skipped}"
                                             if skipped else ""))
    if share > 0.30:
        console.warn(
            "фона больше 30%: модель склонна занижать уверенность и пропускать "
            "объекты. Обычно держат 1–10% — ограничьте dataset.negatives.max_share "
            "или проверьте, не помечены ли пустыми кадры с объектом"
        )
    elif counters["empty"] == 0:
        console.warn("в датасете нет ни одного фонового кадра — будут лишние "
                     "срабатывания на похожих текстурах; добавьте 1–10% фона")


def _resolve_assignment(project: Project, groups: list[str], val_share: float,
                        test_share: float, grouping: str,
                        reset: bool) -> tuple[dict[str, str], str]:
    """Раскладка групп по сплитам с учётом прошлых сборок.

    «Липкие» сплиты (`dataset.stable_splits`) держат уже известные группы на
    своих местах: без этого добор данных перетасовывает train и val, метрики
    соседних прогонов становятся несравнимыми, а дообучение с прошлых весов
    даёт утечку — модель уже видела то, что теперь в val.
    """
    state_path = project.paths.workspace / "splits.json"
    if reset and state_path.exists():
        state_path.unlink()
        console.warn("--reset-splits: прежняя раскладка сплитов удалена")

    stable = bool(project.get("dataset.stable_splits", True))
    if not stable:
        assignment = assign_splits(groups, val_share, test_share)
        return assignment, "сплит пересчитан с нуля (stable_splits: false)"

    state = load_state(state_path)
    previous = state.get("assignment", {}) if state else {}
    if previous and state.get("group_by") != grouping:
        console.warn(f"group_by изменился ({state.get('group_by')} -> {grouping}) — "
                     "прежняя раскладка не применима, считаю заново")
        previous = {}
    if previous and (state.get("val_ratio") != val_share
                     or state.get("test_ratio") != test_share):
        console.warn("доли val/test изменились: к известным группам они не "
                     "применяются, новые доберут пропорции. Полный пересчёт — "
                     "--reset-splits")

    assignment, fresh = extend_assignment(previous, groups, val_share, test_share)
    save_state(state_path, assignment, grouping, val_share, test_share)
    if not previous:
        return assignment, f"раскладка сплитов создана: {state_path.name}"
    return assignment, (f"сплиты сохранены с прошлой сборки, новых групп: {fresh}")


def _write_index(root: Path, pairs: list[Pair], assignment: dict[str, str],
                 skipped: set[str]) -> None:
    """CSV «кадр в датасете -> исходный файл»: чтобы править разметку в источнике."""
    import csv

    with (root / "index.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "split", "group", "source_image", "source_label"])
        for pair in pairs:
            if pair.key in skipped:
                continue
            writer.writerow([pair.key, assignment[pair.group], pair.group,
                             str(pair.image), str(pair.label or "")])


def _collapse_mode(project: Project, classes: list[str]) -> bool:
    """Сводить ли все классы источников в один.

    По умолчанию — да для одноклассовой задачи (детектор региона) и нет,
    когда классов несколько: иначе многоклассовая разметка молча схлопнется.
    """
    configured = project.get("dataset.collapse_classes", "auto")
    if isinstance(configured, bool):
        return configured
    if str(configured).lower() == "auto":
        return len(classes) == 1
    raise SystemExit("dataset.collapse_classes: ожидается auto, true или false")


def _scan_classes(pairs: list[Pair], mappings: dict, keep_classes: set[int] | None,
                  collapse: bool, classes: list[str]) -> tuple[Counter, list[str]]:
    """Пройти разметку до записи: histogram классов и классы вне диапазона."""
    histogram: Counter = Counter()
    problems: list[str] = []
    for pair in pairs:
        if not pair.label:
            continue
        _, _, counts, out_of_range = parse_label(
            pair.label, keep_classes, mappings.get(pair.prefix), collapse, len(classes)
        )
        histogram.update(counts)
        if out_of_range:
            problems.append(
                f"{pair.label.name}: классы {sorted(out_of_range)} вне "
                f"0..{len(classes) - 1}"
            )
    return histogram, problems


def _report_classes(histogram: Counter, classes: list[str]) -> None:
    """Сколько боксов на класс — и предупредить о пустых и о перекосе."""
    if len(classes) == 1:
        return
    console.rule("классы")
    total = sum(histogram.values()) or 1
    for index, name in enumerate(classes):
        count = histogram.get(index, 0)
        console.kv(f"{index} {name}", f"{count} боксов ({100 * count / total:.1f}%)")
    missing = [name for index, name in enumerate(classes) if not histogram.get(index)]
    if missing:
        console.err(f"без единого бокса: {', '.join(missing)} — модель этому "
                    "классу не научится")
    counts = [histogram.get(index, 0) for index in range(len(classes))]
    if min(counts) and max(counts) / min(counts) > 10:
        console.warn(f"перекос классов {max(counts)}:{min(counts)} — редкий класс "
                     "будет распознаваться заметно хуже")


def _warn_about_leakage(pairs: list[Pair], groups: set[str], grouping: str) -> None:
    """Предупредить, когда сплит по кадрам разведёт копии одного снимка."""
    if grouping != "stem":
        return
    augmented = looks_augmented(pairs)
    if augmented > len(pairs) * 0.05:
        console.warn(
            f"{augmented} кадров похожи на аугментированные копии (Roboflow). "
            "При group_by: stem копии одного снимка попадут и в train, и в val — "
            "метрики будут завышены. Поставьте dataset.group_by: roboflow"
        )
    elif len(groups) < len(pairs):
        console.warn("Группы совпадают с кадрами: проверьте dataset.group_by, "
                     "если в данных есть дубликаты или серии одного объекта")


def _write_manifest(project: Project, root: Path, per_source: list[dict],
                    counters: dict, groups: set[str], grouping: str,
                    val_share: float, test_share: float, collapse: bool,
                    histogram: Counter, dropped_negatives: int,
                    split_signature: str) -> Path:
    """Паспорт сборки: по нему видно, на каких данных обучен прогон."""
    config_text = project.paths.config_file.read_text(encoding="utf-8")
    document = {
        "project": project.name,
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mlkit": __version__,
        "classes": project.classes,
        "group_by": grouping,
        "collapse_classes": collapse,
        "split_signature": split_signature,
        "val_ratio": val_share,
        "test_ratio": test_share,
        "sources": per_source,
        "groups": len(groups),
        "counts": {key: counters[key] for key in
                   (*SPLITS, "boxes", "empty", "dropped") if key in counters},
        "boxes_per_class": {project.classes[index]: histogram.get(index, 0)
                            for index in range(len(project.classes))},
        "negatives": {
            "kept": counters.get("empty", 0),
            "dropped_over_limit": dropped_negatives,
            "share": round(counters.get("empty", 0)
                           / max(1, sum(counters[s] for s in SPLITS if s in counters)), 4),
        },
        "config_sha1": hashlib.sha1(config_text.encode("utf-8")).hexdigest()[:12],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


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
        for split in SPLITS
        if (root / split / "images").is_dir()
    }


def dataset_stats(project: Project) -> int:
    """Подробности по собранному датасету: боксы, размеры, утечка между сплитами."""
    root = project.paths.dataset
    if not project.paths.data_yaml.is_file():
        raise SystemExit(f"Датасет не собран — mlkit dataset build {project.name}")

    console.step(f"Датасет «{project.name}»: {root}")
    names = project.classes
    manifest = root / "manifest.json"
    if manifest.is_file():
        document = json.loads(manifest.read_text(encoding="utf-8"))
        console.kv("собран", document.get("built_at"))
        console.kv("group_by", document.get("group_by"))
        console.kv("групп", document.get("groups"))

    total_boxes = 0
    for split in SPLITS:
        labels_dir = root / split / "labels"
        if not labels_dir.is_dir():
            continue
        labels = sorted(labels_dir.glob("*.txt"))
        boxes = empty = 0
        areas: list[float] = []
        per_class: Counter = Counter()
        for label in labels:
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line]
            boxes += len(lines)
            empty += 0 if lines else 1
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    per_class[int(float(parts[0]))] += 1
                    areas.append(float(parts[3]) * float(parts[4]))
        total_boxes += boxes
        console.rule(split)
        console.kv("изображений", len(labels))
        console.kv("боксов", boxes)
        console.kv("негативных (фон)", f"{empty} ({100 * empty / max(1, len(labels)):.0f}%)")
        if len(names) > 1:
            for index, name in enumerate(names):
                console.kv(f"  класс {index} {name}", per_class.get(index, 0))
        if areas:
            areas.sort()
            console.kv("медианная площадь бокса", f"{areas[len(areas) // 2] * 100:.2f}% кадра")
            console.kv("самый мелкий", f"{areas[0] * 100:.3f}% кадра")

    console.info("")
    console.kv("итого боксов", total_boxes)
    _report_leakage(root)
    return 0


def _report_leakage(root: Path) -> None:
    """Проверить, не попали ли копии одного снимка в разные сплиты."""
    from .discover import ROBOFLOW_SUFFIX

    bases: dict[str, set[str]] = {}
    for split in SPLITS:
        images_dir = root / split / "images"
        if not images_dir.is_dir():
            continue
        for image in images_dir.iterdir():
            base = ROBOFLOW_SUFFIX.sub("", image.stem)
            bases.setdefault(base, set()).add(split)

    shared = [base for base, splits in bases.items() if len(splits) > 1]
    console.rule("утечка между сплитами")
    if shared:
        console.err(f"кадров с копиями в разных сплитах: {len(shared)} "
                    f"из {len(bases)} — метрики будут завышены")
        console.info("  лечится параметром dataset.group_by (roboflow / parent / regex:…)")
    else:
        console.ok("копий одного снимка в разных сплитах нет")
