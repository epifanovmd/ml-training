"""Загрузка конфигурации проекта: defaults ядра + projects/<имя>/project.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECTS_DIR, REPO_ROOT, ProjectPaths, resolve

DEFAULTS_FILE = Path(__file__).resolve().parent / "defaults.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Слияние словарей: значения override побеждают, вложенность сохраняется."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def list_projects() -> list[str]:
    if not PROJECTS_DIR.is_dir():
        return []
    return sorted(
        p.name for p in PROJECTS_DIR.iterdir()
        if p.is_dir() and (p / "project.yaml").is_file()
    )


@dataclass
class Project:
    """Проект = конфиг + каталоги. Ядро работает только с этим объектом."""

    name: str
    data: dict = field(repr=False)
    paths: ProjectPaths

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted)
        if value is None:
            raise SystemExit(f"В {self.paths.config_file} не задано «{dotted}»")
        return value

    @property
    def classes(self) -> list[str]:
        classes = self.get("classes", [])
        if not classes:
            raise SystemExit(f"В {self.paths.config_file} пустой список classes")
        return list(classes)

    def project_file(self, relative: str) -> Path:
        """Путь к файлу проекта — относительно projects/<имя>/."""
        candidate = Path(relative).expanduser()
        if candidate.is_absolute():
            return candidate
        return self.paths.project_dir / candidate


def load_project(name: str) -> Project:
    paths = ProjectPaths(name=name)
    if not paths.config_file.is_file():
        available = ", ".join(list_projects()) or "нет ни одного"
        raise SystemExit(
            f"Нет проекта «{name}» ({paths.config_file}). Доступные: {available}"
        )
    data = deep_merge(read_yaml(DEFAULTS_FILE), read_yaml(paths.config_file))
    data["name"] = name
    return Project(name=name, data=data, paths=paths)


def dataset_sources(project: Project) -> list[Path]:
    """Каталоги-источники размеченных данных для сборки датасета.

    По умолчанию берутся все подкаталоги datasets/<проект>/.
    """
    from .paths import DATASETS_DIR

    configured = project.get("dataset.sources", []) or []
    if configured:
        found: list[Path] = []
        for pattern in configured:
            pattern = str(pattern)
            if any(ch in pattern for ch in "*?["):
                found.extend(sorted(resolve(".").glob(pattern)))
            else:
                found.append(resolve(pattern))
        return [p for p in found if p.exists()]

    ready = DATASETS_DIR / project.name
    if not ready.is_dir():
        return []
    return sorted(path for path in ready.iterdir() if path.is_dir())
