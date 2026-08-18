"""Единая точка входа: mlkit <команда> <проект> [опции].

Каждая стадия — отдельная команда; все они работают с одним описанием
проекта (projects/<имя>/project.yaml) и одной раскладкой каталогов
(workspace/<имя>/…). Размеченные данные приходят из datasets/<проект>/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import __version__, console
from .config import dataset_sources, list_projects, load_project
from .paths import REPO_ROOT

EXPORT_PYTHON = REPO_ROOT / ".venv-export" / "bin" / "python"


# --------------------------------------------------------------------------
# команды
# --------------------------------------------------------------------------
def cmd_projects(args: argparse.Namespace) -> int:
    projects = list_projects()
    if not projects:
        console.warn("Ни одного проекта в projects/")
        return 1
    console.step("Проекты")
    for name in projects:
        project = load_project(name)
        console.kv(name, project.get("description", "") or ", ".join(project.classes))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .dataset import dataset_status
    from .exporting import export_status

    project = load_project(args.project)
    console.step(f"Проект «{project.name}»")
    console.kv("конфиг", project.paths.config_file)
    console.kv("классы", ", ".join(project.classes))
    console.kv("рабочий каталог", project.paths.workspace)

    console.rule("1. датасет")
    sources = dataset_sources(project)
    for source in sources:
        console.kv(str(source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT)
                       else source), "источник")
    if not sources:
        console.info(f"  нет источников — положите датасет в datasets/{project.name}/")
    counts = dataset_status(project)
    if counts:
        console.kv("train / val", f"{counts.get('train', 0)} / {counts.get('val', 0)}")
    else:
        console.info("  не собран")

    console.rule("2. обучение")
    runs = sorted(p.name for p in project.paths.runs.glob("*")
                  if (p / "weights" / "best.pt").is_file()) \
        if project.paths.runs.is_dir() else []
    console.info("  " + (", ".join(runs) if runs else "прогонов нет"))

    console.rule("3. экспорт")
    artifacts = export_status(project)
    console.info("  " + (", ".join(artifacts) if artifacts else "нет"))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    console.step(f"Проверка окружения (mlkit {__version__})")
    console.kv("python", f"{sys.version.split()[0]} ({sys.executable})")
    ok = True

    try:
        import ultralytics

        console.kv("ultralytics", ultralytics.__version__)
        import torch

        console.kv("torch", torch.__version__)
        console.kv("mps доступен", torch.backends.mps.is_available())
        console.kv("cuda доступна", torch.cuda.is_available())
    except ImportError as error:
        console.err(f"обучающее окружение не готово: {error} → make install")
        ok = False

    console.kv(".venv-export", "есть" if EXPORT_PYTHON.is_file()
               else "нет → make install-export")

    if args.project:
        project = load_project(args.project)
        console.rule(project.name)
        sources = dataset_sources(project)
        console.kv("источники данных", ", ".join(p.name for p in sources) or "нет")
        console.kv("датасет собран", "да" if project.paths.data_yaml.is_file() else "нет")
    return 0 if ok else 1


def cmd_dataset(args: argparse.Namespace) -> int:
    from .dataset import build_dataset

    project = load_project(args.project)
    return build_dataset(project, sources=args.source, verify=args.verify,
                         val_ratio=args.val_ratio)


def cmd_dataset_stats(args: argparse.Namespace) -> int:
    from .dataset import dataset_stats

    return dataset_stats(load_project(args.project))


def cmd_train(args: argparse.Namespace) -> int:
    from .training import train

    project = load_project(args.project)
    return train(project, epochs=args.epochs, device=args.device, batch=args.batch,
                 base_model=args.model, imgsz=args.imgsz, name=args.name,
                 resume=args.resume)


def cmd_val(args: argparse.Namespace) -> int:
    from .training import validate

    return validate(load_project(args.project), run=args.run, device=args.device)


def cmd_predict(args: argparse.Namespace) -> int:
    from .evaluate import predict

    project = load_project(args.project)
    return predict(project, input_dir=args.input, output_dir=args.output,
                   weights=args.weights, run=args.run, conf=args.conf,
                   device=args.device, show=args.show)


def cmd_export(args: argparse.Namespace) -> int:
    from .exporting import export

    if args.pretrained:
        return export(None, pretrained=args.pretrained, formats=args.format,
                      out=args.out, imgsz=args.imgsz)
    project = load_project(args.project)
    return export(project, run=args.run, weights=args.weights, formats=args.format,
                  out=args.out, imgsz=args.imgsz)


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Последовательный прогон стадий. Экспорт уходит в .venv-export."""
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    known = {"dataset", "train", "export"}
    unknown = set(stages) - known
    if unknown:
        raise SystemExit(f"Неизвестные стадии: {', '.join(sorted(unknown))}")

    for stage in stages:
        console.rule(f"стадия: {stage}")
        if stage == "dataset":
            code = cmd_dataset(argparse.Namespace(
                project=args.project, source=None, verify=False, val_ratio=None))
        elif stage == "train":
            code = cmd_train(argparse.Namespace(
                project=args.project, epochs=args.epochs, device=args.device,
                batch=None, model=None, imgsz=None, name=args.name, resume=False))
        else:
            code = _export_in_export_venv(args)
        if code != 0:
            console.err(f"Стадия «{stage}» завершилась с кодом {code}")
            return code
    console.ok("Конвейер пройден полностью")
    return 0


def _export_in_export_venv(args: argparse.Namespace) -> int:
    """Экспорт требует .venv-export — при необходимости перезапускаемся в нём."""
    if Path(sys.executable).resolve() == EXPORT_PYTHON.resolve():
        return cmd_export(argparse.Namespace(
            project=args.project, run=args.name, weights=None, format=None,
            out=None, pretrained=None, imgsz=None))
    if not EXPORT_PYTHON.is_file():
        console.err(f"Нет {EXPORT_PYTHON} — выполните make install-export")
        return 1
    command = [str(EXPORT_PYTHON), "-m", "mlkit", "export", args.project]
    if args.name:
        command += ["--run", args.name]
    environment = {**os.environ,
                   "PYTHONPATH": os.pathsep.join(
                       [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)}
    console.info(f"  запускаю: {' '.join(command)}")
    return subprocess.run(command, cwd=REPO_ROOT, env=environment).returncode


# --------------------------------------------------------------------------
# разбор аргументов
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlkit", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"mlkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="команда")

    def project_arg(command: argparse.ArgumentParser) -> None:
        command.add_argument("project", help=f"проект из projects/ "
                                             f"({', '.join(list_projects()) or 'нет'})")

    p = sub.add_parser("projects", help="список проектов")
    p.set_defaults(func=cmd_projects)

    p = sub.add_parser("status", help="состояние всех стадий проекта")
    project_arg(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("doctor", help="проверить окружение и данные проекта")
    p.add_argument("project", nargs="?", help="проверить ещё и конкретный проект")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("dataset", help="1) собрать train/val из размеченных источников")
    project_arg(p)
    p.add_argument("--source", action="append", help="источник (можно повторять)")
    p.add_argument("--verify", action="store_true", help="только проверить, не записывать")
    p.add_argument("--val-ratio", type=float, help="доля val")
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("dataset-stats", help="1a) статистика собранного датасета")
    project_arg(p)
    p.set_defaults(func=cmd_dataset_stats)

    p = sub.add_parser("train", help="2) обучить модель")
    project_arg(p)
    p.add_argument("--epochs", type=int)
    p.add_argument("--device", help="mps / cpu / 0")
    p.add_argument("--batch", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--model", help="базовые веса вместо model.base")
    p.add_argument("--name", help="имя прогона в runs/")
    p.add_argument("--resume", action="store_true", help="продолжить прерванное обучение")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("val", help="2a) метрики обученной модели на val")
    project_arg(p)
    p.add_argument("--run", help="какой прогон валидировать")
    p.add_argument("--device")
    p.set_defaults(func=cmd_val)

    p = sub.add_parser("predict", help="2b) прогнать модель по своим картинкам")
    project_arg(p)
    p.add_argument("--input", help="каталог с картинками (по умолчанию samples/)")
    p.add_argument("--output", help="куда класть аннотированные копии")
    p.add_argument("--weights", help="конкретные веса")
    p.add_argument("--run", help="прогон, чей best.pt взять")
    p.add_argument("--conf", type=float)
    p.add_argument("--device")
    p.add_argument("--show", action="store_true", help="открыть результат (macOS)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("export", help="3) конвертировать в CoreML/TFLite")
    p.add_argument("project", nargs="?", help="проект (не нужен при --pretrained)")
    p.add_argument("--run", help="прогон, чей best.pt экспортировать")
    p.add_argument("--weights", help="конкретные веса")
    p.add_argument("--format", action="append",
                   choices=("coreml", "tflite", "onnx", "torchscript"),
                   help="формат (можно повторять; по умолчанию export.formats)")
    p.add_argument("--out", help="каталог назначения")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--pretrained", help="экспортировать готовую модель, например yolo11n.pt")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("pipeline", help="прогнать несколько стадий подряд")
    project_arg(p)
    p.add_argument("--stages", default="dataset,train,export",
                   help="через запятую: dataset,train,export")
    p.add_argument("--epochs", type=int)
    p.add_argument("--device")
    p.add_argument("--name", help="имя прогона обучения")
    p.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.warn("Прервано пользователем")
        return 130
    except SystemExit as error:
        if isinstance(error.code, str):
            console.err(error.code)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
