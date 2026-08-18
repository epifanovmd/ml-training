"""Раскладка каталогов. Единственное место, где зашиты пути."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# src/mlkit/paths.py -> src/mlkit -> src -> корень репозитория
REPO_ROOT = Path(__file__).resolve().parents[2]

PROJECTS_DIR = REPO_ROOT / "projects"
# Размеченные датасеты в YOLO-формате: каждый подкаталог — отдельный источник
DATASETS_DIR = REPO_ROOT / "datasets"
WORKSPACE_DIR = REPO_ROOT / "workspace"    # всё генерируемое
WEIGHTS_DIR = REPO_ROOT / "weights"        # кэш базовых чекпойнтов
PRETRAINED_DIR = WORKSPACE_DIR / "_pretrained"


def resolve(path: str | Path) -> Path:
    """Путь из конфига: относительные считаются от корня репозитория."""
    candidate = Path(str(path)).expanduser()
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """Все каталоги одного проекта."""

    name: str

    @property
    def project_dir(self) -> Path:
        return PROJECTS_DIR / self.name

    @property
    def config_file(self) -> Path:
        return self.project_dir / "project.yaml"

    @property
    def samples(self) -> Path:
        return self.project_dir / "samples"

    @property
    def workspace(self) -> Path:
        return WORKSPACE_DIR / self.name

    @property
    def dataset(self) -> Path:
        """Собранный train/val + data.yaml."""
        return self.workspace / "dataset"

    @property
    def data_yaml(self) -> Path:
        return self.dataset / "data.yaml"

    @property
    def runs(self) -> Path:
        return self.workspace / "runs"

    @property
    def exports(self) -> Path:
        return self.workspace / "exports"

    @property
    def previews(self) -> Path:
        """Результаты predict — для глазной проверки модели."""
        return self.workspace / "previews"

    def run_dir(self, run: str | None = None) -> Path:
        return self.runs / (run or self.name)

    def weights(self, run: str | None = None) -> Path:
        return self.run_dir(run) / "weights" / "best.pt"

    def ensure(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
