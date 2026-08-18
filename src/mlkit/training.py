"""Стадия 2 — обучение.

Все параметры приходят из project.yaml: секция `train` передаётся в
ultralytics как есть, `augment` — тоже, а `profiles` позволяет держать
рядом несколько наборов настроек (быстрая проверка / выжимание качества)
и переключаться одним флагом.

Перед стартом выполняется предполётная проверка: она ловит то, что иначе
обнаруживается через несколько часов обучения — пустой val, утечку между
сплитами, батч больше датасета, mosaic, который никогда не выключится.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from . import __version__, console
from .config import Project, deep_merge
from .paths import WEIGHTS_DIR, resolve

# Ключи, которые ultralytics не понимает, он молча игнорирует не всегда —
# лучше поймать опечатку до старта обучения
_IGNORED_IN_CHECK = {"data", "imgsz", "project", "name", "exist_ok", "resume"}


def known_keys() -> set[str]:
    from ultralytics.cfg import DEFAULT_CFG_DICT

    return set(DEFAULT_CFG_DICT)


def validate_keys(section: str, values: dict) -> None:
    """Проверить имена параметров: опечатка молча меняет поведение обучения."""
    valid = known_keys()
    for key in values:
        if key in valid or key in _IGNORED_IN_CHECK:
            continue
        hint = difflib.get_close_matches(key, sorted(valid), n=3)
        raise SystemExit(
            f"В секции «{section}» неизвестный параметр «{key}»"
            + (f". Возможно: {', '.join(hint)}" if hint else "")
        )


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


def run_name_for(project: Project, name: str | None, base_model: str | None,
                 profile: str | None) -> str:
    if name:
        return name
    parts = [project.name]
    previous = _run_of_weights(project, Path(base_model)) if base_model else None
    if previous:
        # Дообучение поверх своего прогона: в имени полезен исходный прогон,
        # а не бесполезное «best». Профиль finetune в имени не дублируем.
        short = previous[len(project.name) + 1:] if previous.startswith(f"{project.name}-") \
            else (previous if previous != project.name else "")
        parts.append(f"ft-{short}" if short else "ft")
        if profile and profile != "finetune":
            parts.append(profile)
    elif base_model:
        parts.append(Path(base_model).stem)
        if profile:
            parts.append(profile)
    elif profile:
        parts.append(profile)
    return "-".join(parts)


def _run_of_weights(project: Project, weights: Path) -> str | None:
    """Имя прогона, если веса лежат внутри runs/ этого проекта."""
    try:
        relative = weights.resolve().relative_to(project.paths.runs.resolve())
    except (ValueError, OSError):
        return None
    return relative.parts[0] if relative.parts else None


def weights_of_run(project: Project, run: str) -> str:
    """Путь к best.pt конкретного прогона — для дообучения поверх него."""
    weights = project.paths.weights(run)
    if not weights.is_file():
        available = sorted(path.name for path in project.paths.runs.glob("*")
                           if (path / "weights" / "best.pt").is_file()) \
            if project.paths.runs.is_dir() else []
        raise SystemExit(
            f"Нет весов прогона «{run}» ({weights}). "
            f"Доступные: {', '.join(available) or 'нет ни одного'}"
        )
    return str(weights)


def apply_profile(project: Project, profile: str | None) -> dict:
    """Наложить профиль из `profiles.<имя>` поверх основных секций."""
    data = project.data
    if not profile:
        return data
    profiles = project.get("profiles", {}) or {}
    if profile not in profiles:
        available = ", ".join(sorted(profiles)) or "нет ни одного"
        raise SystemExit(f"Нет профиля «{profile}». Доступные: {available}")
    console.kv("профиль", profile)
    return deep_merge(data, profiles[profile] or {})


def auto_device() -> str:
    """Лучшее доступное устройство, если в конфиге указано device: auto."""
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_auto(settings: dict) -> dict:
    """Развернуть значения «auto» в конкретные числа для текущей машины."""
    settings = dict(settings)
    device = str(settings.get("device", "")).lower()
    if device in ("auto", ""):
        settings["device"] = auto_device()

    if str(settings.get("workers", "")).lower() == "auto":
        settings["workers"] = max(2, min(8, (os.cpu_count() or 4) // 2))

    if str(settings.get("batch", "")).lower() == "auto":
        # -1 = автоподбор по свободной видеопамяти, работает только на CUDA
        if str(settings["device"]).isdigit():
            settings["batch"] = -1
        else:
            settings["batch"] = 16
            console.warn("batch: auto поддерживается только на CUDA — взял 16")
    return settings


def _split_counts(dataset: Path) -> dict[str, int]:
    counts = {}
    for split in ("train", "val", "test"):
        images = dataset / split / "images"
        if images.is_dir():
            counts[split] = sum(1 for _ in images.iterdir())
    return counts


def _leaked_frames(dataset: Path) -> int:
    """Сколько кадров имеют копию-аугментацию в другом сплите."""
    from .dataset.discover import ROBOFLOW_SUFFIX

    seen: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        images = dataset / split / "images"
        if not images.is_dir():
            continue
        for image in images.iterdir():
            seen.setdefault(ROBOFLOW_SUFFIX.sub("", image.stem), set()).add(split)
    return sum(1 for splits in seen.values() if len(splits) > 1)


def preflight(project: Project, settings: dict, imgsz: int) -> None:
    """Проверки перед стартом: дешёвые сейчас, дорогие — через три часа."""
    dataset = project.paths.dataset
    counts = _split_counts(dataset)
    console.rule("проверка перед стартом")
    console.kv("кадров train / val", f"{counts.get('train', 0)} / {counts.get('val', 0)}")

    if not counts.get("train"):
        raise SystemExit("В train нет ни одного кадра — пересоберите датасет")
    if not counts.get("val"):
        console.err("val пуст: не будет ни метрик, ни ранней остановки, "
                    "а best.pt выберется наугад")
    elif counts["val"] < 30:
        console.warn(f"val крошечный ({counts['val']} кадров) — метрики будут шумными")
    if counts["train"] < 200:
        console.warn(f"мало данных для обучения с нуля ({counts['train']} кадров): "
                     "ждите переобучения, помогут freeze и меньшее число эпох")

    leaked = _leaked_frames(dataset)
    if leaked:
        console.err(f"{leaked} кадров имеют копии в разных сплитах — метрики "
                    "будут завышены (см. dataset.group_by)")

    if imgsz % 32:
        console.warn(f"imgsz={imgsz} не кратен 32 — ultralytics округлит сам")

    epochs = int(settings.get("epochs", 0) or 0)
    batch = settings.get("batch")
    if isinstance(batch, int) and batch > 0 and batch > counts["train"]:
        console.warn(f"batch={batch} больше train ({counts['train']}) — "
                     "будет один неполный батч на эпоху")
    close_mosaic = int(settings.get("close_mosaic", 0) or 0)
    if close_mosaic and epochs and close_mosaic >= epochs:
        console.warn(f"close_mosaic={close_mosaic} ≥ epochs={epochs}: "
                     "мозаика выключится сразу и обучение пойдёт без неё")
    warmup = float(settings.get("warmup_epochs", 0) or 0)
    if epochs and warmup >= epochs:
        console.warn(f"warmup_epochs={warmup} ≥ epochs={epochs}")
    fraction = float(settings.get("fraction", 1.0) or 1.0)
    if fraction < 1.0:
        console.warn(f"fraction={fraction}: обучение идёт лишь на части данных")

    labels = dataset / "train" / "labels"
    if labels.is_dir():
        files = list(labels.glob("*.txt"))
        empty = sum(1 for path in files if not path.read_text(encoding="utf-8").strip())
        if files and empty / len(files) > 0.3:
            console.warn(
                f"фоновых кадров {empty} из {len(files)} "
                f"({100 * empty / len(files):.0f}%): ориентир 1–10%, иначе модель "
                "занижает уверенность. Ограничьте dataset.negatives.max_share"
            )
        elif files and not empty:
            console.warn("фоновых кадров нет — будут лишние срабатывания на "
                         "похожих текстурах; полезно добавить 1–10%")

    _check_classes(project, dataset)


def _check_classes(project: Project, dataset: Path) -> None:
    """При нескольких классах — предупредить о пустых и перекошенных."""
    from collections import Counter

    classes = project.classes
    if len(classes) < 2:
        return
    for split in ("train", "val"):
        labels = dataset / split / "labels"
        if not labels.is_dir():
            continue
        counts: Counter = Counter()
        for path in labels.glob("*.txt"):
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    counts[int(float(parts[0]))] += 1
        missing = [name for index, name in enumerate(classes) if not counts.get(index)]
        if missing:
            console.err(f"в {split} нет боксов классов: {', '.join(missing)}")
        values = [counts.get(index, 0) for index in range(len(classes))]
        if min(values) and max(values) / min(values) > 10:
            console.warn(f"в {split} перекос классов {max(values)}:{min(values)} — "
                         "редкий класс будет хуже; помогут добор данных или "
                         "увеличение cls в train")


def train(project: Project, epochs: int | None = None, device: str | None = None,
          batch: int | None = None, base_model: str | None = None,
          imgsz: int | None = None, name: str | None = None,
          resume: bool = False, profile: str | None = None,
          overrides: dict | None = None, dry_run: bool = False,
          from_run: str | None = None) -> int:
    from ultralytics import YOLO

    data_yaml = project.paths.data_yaml
    if not data_yaml.is_file():
        raise SystemExit(
            f"Нет {data_yaml} — сначала соберите датасет: mlkit dataset {project.name}"
        )

    console.step(f"Обучение «{project.name}»")
    data = apply_profile(project, profile)
    settings = dict(data.get("train", {}) or {})
    augment = dict(data.get("augment", {}) or {})
    size = int(imgsz or data.get("model", {}).get("imgsz", 640))

    if epochs is not None:
        settings["epochs"] = epochs
    if device is not None:
        settings["device"] = device
    if batch is not None:
        settings["batch"] = batch
    settings.update(overrides or {})

    validate_keys("train", settings)
    validate_keys("augment", augment)
    settings = resolve_auto(settings)

    if from_run:
        base_model = weights_of_run(project, from_run)
    weights = resolve_base_weights(
        base_model or data.get("model", {}).get("base", "yolo11n.pt")
    )
    run = run_name_for(project, name, base_model, profile)
    if resume:
        last = project.paths.run_dir(run) / "weights" / "last.pt"
        if not last.is_file():
            raise SystemExit(f"Нечего продолжать: нет {last}")
        weights = str(last)

    console.kv("прогон", f"runs/{run}")
    if not resume and (project.paths.run_dir(run) / "weights" / "best.pt").is_file():
        console.warn(f"прогон «{run}» уже существует и будет перезаписан — "
                     "задайте NAME=<имя>, если он ещё нужен")
    console.kv("датасет", data_yaml)
    console.kv("базовые веса", weights)
    console.kv("imgsz", size)
    console.kv("эпох", settings.get("epochs"))
    console.kv("batch / workers", f"{settings.get('batch')} / {settings.get('workers')}")
    console.kv("устройство", settings.get("device"))
    console.kv("оптимизатор", f"{settings.get('optimizer')} lr0={settings.get('lr0')}")

    preflight(project, settings, size)
    _warn_if_split_changed(project, weights)
    if dry_run:
        console.warn("--dry-run: параметры проверены, обучение не запускалось")
        console.info(json.dumps({"train": settings, "augment": augment},
                                ensure_ascii=False, indent=2, default=str))
        return 0

    # Паспорт пишем до старта: прогон может прерваться, а знать, на чём он
    # обучался, нужно и тогда
    _record_run(project, run, weights, size, settings, augment, profile)

    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml),
        imgsz=size,
        project=str(project.paths.runs),
        name=run,
        exist_ok=True,
        resume=resume,
        **settings,
        **augment,
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
    console.info(f"  дальше: mlkit errors {project.name}" +
                 (f" --run {run}" if run != project.name else "") +
                 "  (посмотреть, где модель ошибается)")
    hint = f"mlkit export {project.name}"
    if run != project.name:
        hint += f" --run {run}"
    console.info(f"  затем: {hint}")
    return 0


def _warn_if_split_changed(project: Project, weights: str) -> None:
    """Сравнить сплит текущего датасета со сплитом прогона, с которого стартуем.

    Если раскладка изменилась, прошлая модель могла обучаться на кадрах,
    которые теперь в val: метрики окажутся завышенными, а сравнение с тем
    прогоном — некорректным.
    """
    previous_run = _run_of_weights(project, Path(weights))
    if not previous_run:
        return
    old = project.paths.run_dir(previous_run) / "dataset_manifest.json"
    new = project.paths.dataset / "manifest.json"
    if not (old.is_file() and new.is_file()):
        return
    try:
        old_signature = json.loads(old.read_text(encoding="utf-8")).get("split_signature")
        new_signature = json.loads(new.read_text(encoding="utf-8")).get("split_signature")
    except json.JSONDecodeError:
        return
    if old_signature and new_signature and old_signature != new_signature:
        console.warn(
            f"сплит изменился с прогона «{previous_run}» ({old_signature} -> "
            f"{new_signature}): часть кадров могла переехать из train в val, "
            "метрики будут завышены. Помогает dataset.stable_splits: true"
        )


def _record_run(project: Project, run: str, weights: str, imgsz: int,
                settings: dict, augment: dict, profile: str | None) -> None:
    """Паспорт прогона: на каких данных и чем обучено.

    Без него по каталогу прогона нельзя восстановить состав датасета —
    а он меняется при каждом пополнении данных.
    """
    run_dir = project.paths.run_dir(run)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = project.paths.dataset / "manifest.json"
    if manifest.is_file():
        shutil.copy2(manifest, run_dir / "dataset_manifest.json")

    versions = {"mlkit": __version__}
    for name in ("ultralytics", "torch", "torchvision", "numpy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "нет"

    document = {
        "project": project.name,
        "run": run,
        "profile": profile,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_weights": str(weights),
        "imgsz": imgsz,
        "train": settings,
        "augment": augment,
        "versions": versions,
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def validate(project: Project, run: str | None = None, device: str | None = None,
             split: str = "val") -> int:
    """Посчитать метрики обученной модели на val- или test-сплите."""
    from ultralytics import YOLO

    weights = project.paths.weights(run)
    if not weights.is_file():
        raise SystemExit(f"Нет весов {weights}")
    if split == "test" and not (project.paths.dataset / "test" / "images").is_dir():
        raise SystemExit(
            "Нет test-сплита. Соберите датасет с dataset.test_ratio > 0:\n"
            f"  mlkit dataset {project.name} --test-ratio 0.15"
        )

    console.step(f"Метрики {weights.name} на сплите {split}")
    model = YOLO(str(weights))
    metrics = model.val(data=str(project.paths.data_yaml),
                        imgsz=int(project.get("model.imgsz", 640)),
                        split=split,
                        project=str(project.paths.runs),
                        name=f"{run or project.name}/{split}", exist_ok=True,
                        **({"device": device} if device else {}))
    console.kv("mAP50", f"{metrics.box.map50:.4f}")
    console.kv("mAP50-95", f"{metrics.box.map:.4f}")
    console.kv("precision", f"{metrics.box.mp:.4f}")
    console.kv("recall", f"{metrics.box.mr:.4f}")
    if split == "val":
        console.info("  вес выбран по этому же сплиту — для честной цифры "
                     "используйте test (mlkit dataset … --test-ratio 0.15)")
    return 0


def tune(project: Project, iterations: int = 20, epochs: int = 15,
         profile: str | None = None, device: str | None = None,
         base_model: str | None = None) -> int:
    """Генетический подбор гиперпараметров ultralytics.

    Каждая итерация — короткое обучение, поэтому запускать имеет смысл
    на устоявшемся датасете и с небольшим epochs. Найденное кладётся
    в runs/<прогон>-tune/best_hyperparameters.yaml — оттуда значения
    переносятся в project.yaml руками, осознанно.
    """
    from ultralytics import YOLO

    if not project.paths.data_yaml.is_file():
        raise SystemExit(f"Нет {project.paths.data_yaml} — сначала соберите датасет")

    data = apply_profile(project, profile)
    settings = resolve_auto(dict(data.get("train", {}) or {}))
    if device:
        settings["device"] = device
    size = int(data.get("model", {}).get("imgsz", 640))
    weights = resolve_base_weights(
        base_model or data.get("model", {}).get("base", "yolo11n.pt"))
    run = f"{project.name}-tune"

    console.step(f"Подбор гиперпараметров: {iterations} итераций по {epochs} эпох")
    console.kv("оценка времени", f"≈ {iterations} × {epochs} эпох обучения")
    console.kv("результат", project.paths.runs / run / "best_hyperparameters.yaml")

    model = YOLO(weights)
    model.tune(
        data=str(project.paths.data_yaml),
        imgsz=size,
        epochs=epochs,
        iterations=iterations,
        optimizer=settings.get("optimizer", "auto"),
        device=settings.get("device"),
        batch=settings.get("batch", 16),
        workers=settings.get("workers", 8),
        project=str(project.paths.runs),
        name=run,
        exist_ok=True,
        plots=False,
        save=False,
        val=True,
    )
    console.ok("Готово. Перенесите найденные значения в project.yaml осознанно: "
               "подбор оптимизирует метрику на val, а не устойчивость модели")
    return 0


def list_runs(project: Project) -> int:
    """Таблица прогонов с лучшими метриками — чтобы выбрать, что экспортировать."""
    import csv

    runs_dir = project.paths.runs
    if not runs_dir.is_dir():
        raise SystemExit(f"Нет прогонов в {runs_dir}")

    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        results = run_dir / "results.csv"
        if not results.is_file():
            continue
        best50 = best95 = 0.0
        epochs = 0
        with results.open(encoding="utf-8") as fh:
            for record in csv.DictReader(fh):
                clean = {key.strip(): value for key, value in record.items() if key}
                epochs += 1
                best50 = max(best50, float(clean.get("metrics/mAP50(B)", 0) or 0))
                best95 = max(best95, float(clean.get("metrics/mAP50-95(B)", 0) or 0))
        info = run_dir / "run_info.json"
        base = imgsz = "—"
        if info.is_file():
            document = json.loads(info.read_text(encoding="utf-8"))
            base = Path(str(document.get("base_weights", "—"))).name
            imgsz = document.get("imgsz", "—")
        weights = run_dir / "weights" / "best.pt"
        rows.append({
            "run": run_dir.name, "mAP50": best50, "mAP50-95": best95,
            "epochs": epochs, "base": base, "imgsz": imgsz,
            "weights": "есть" if weights.is_file() else "нет",
        })

    if not rows:
        raise SystemExit(f"В {runs_dir} нет завершённых прогонов")
    rows.sort(key=lambda row: row["mAP50-95"], reverse=True)

    console.step(f"Прогоны «{project.name}» (лучший сверху)")
    console.info(f"  {'прогон':<28} {'mAP50':>7} {'mAP50-95':>9} {'эпох':>6} "
                 f"{'база':>14} {'imgsz':>6}  веса")
    for row in rows:
        console.info(f"  {row['run']:<28} {row['mAP50']:>7.4f} {row['mAP50-95']:>9.4f} "
                     f"{row['epochs']:>6} {str(row['base']):>14} {str(row['imgsz']):>6}  "
                     f"{row['weights']}")
    console.info(f"\n  экспорт выбранного: mlkit export {project.name} --run <прогон>")
    return 0
