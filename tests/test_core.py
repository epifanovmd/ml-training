"""Тесты ядра: разбор разметки, сплит, слияние конфигов.

Запуск: make test (или .venv/bin/python -m unittest discover -s tests).
Сеть и веса не нужны — проверяется только чистая логика.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlkit.config import deep_merge, dataset_sources, load_project  # noqa: E402
from mlkit.dataset.discover import (find_pairs, label_for, parse_label,  # noqa: E402
                                    source_prefix, split_of)


class TestDataset(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "label.txt"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, text: str) -> Path:
        self.path.write_text(text, encoding="utf-8")
        return self.path

    def test_classes_remapped_to_zero(self):
        lines, dropped = parse_label(self.write("3 0.5 0.5 0.2 0.2\n"), None)
        self.assertEqual(lines, ["0 0.500000 0.500000 0.200000 0.200000"])
        self.assertEqual(dropped, 0)

    def test_keep_classes_filters(self):
        lines, _ = parse_label(self.write("0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n"), {2})
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("0 0.100000"))

    def test_out_of_range_dropped(self):
        lines, dropped = parse_label(self.write("0 1.5 0.5 0.2 0.2\n0 0.5 0.5 0 0.2\n"), None)
        self.assertEqual(lines, [])
        self.assertEqual(dropped, 2)

    def test_split_is_deterministic_and_balanced(self):
        names = [f"image-{index}" for index in range(4000)]
        splits = [split_of(name, 0.15) for name in names]
        self.assertEqual(splits, [split_of(name, 0.15) for name in names])
        share = splits.count("val") / len(splits)
        self.assertTrue(0.12 < share < 0.18, f"доля val = {share:.3f}")

    def test_prefix_depends_on_directory_name_only(self):
        self.assertEqual(source_prefix(Path("/a/b/roboflow-a")),
                         source_prefix(Path("/completely/other/roboflow-a")))


class TestDiscovery(unittest.TestCase):
    """Поиск пар «изображение + разметка» в источнике YOLO-формата."""

    def setUp(self):
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "source"
        (self.root / "images").mkdir(parents=True)
        (self.root / "labels").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_pairs_found_in_sibling_labels(self):
        (self.root / "images" / "a.jpg").write_bytes(b"")
        (self.root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        pairs = find_pairs(self.root)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].label.name, "a.txt")

    def test_image_without_label_skipped(self):
        (self.root / "images" / "b.jpg").write_bytes(b"")
        self.assertEqual(find_pairs(self.root), [])

    def test_label_next_to_image(self):
        (self.root / "c.png").write_bytes(b"")
        (self.root / "c.txt").write_text("")
        self.assertIsNotNone(label_for(self.root / "c.png"))


class TestProjects(unittest.TestCase):
    def test_sources_come_from_datasets_dir(self):
        project = load_project("container-code")
        for source in dataset_sources(project):
            self.assertEqual(source.parent.name, "container-code")
            self.assertEqual(source.parent.parent.name, "datasets")

    def test_config_defaults_merged(self):
        project = load_project("plate")
        self.assertEqual(project.classes, ["plate"])
        self.assertIsNotNone(project.get("train.epochs"))
        self.assertEqual(project.get("export.formats"), ["coreml", "tflite"])


class TestConfig(unittest.TestCase):
    def test_deep_merge_keeps_untouched_branches(self):
        base = {"train": {"epochs": 100, "device": "mps"}, "model": {"base": "yolo11n.pt"}}
        merged = deep_merge(base, {"train": {"epochs": 5}})
        self.assertEqual(merged["train"], {"epochs": 5, "device": "mps"})
        self.assertEqual(merged["model"], {"base": "yolo11n.pt"})


if __name__ == "__main__":
    unittest.main()
