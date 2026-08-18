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
from mlkit.dataset.build import _limit_negatives  # noqa: E402
from mlkit.dataset.discover import (find_pairs, group_key, label_for,  # noqa: E402
                                    parse_label, resolve_class_map,
                                    source_class_names, source_prefix)
from mlkit.dataset.splits import (assign_splits, extend_assignment,  # noqa: E402
                                  signature)


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

    def test_classes_collapsed_to_zero(self):
        lines, dropped, counts, _ = parse_label(self.write("3 0.5 0.5 0.2 0.2\n"))
        self.assertEqual(lines, ["0 0.500000 0.500000 0.200000 0.200000"])
        self.assertEqual((dropped, counts[0]), (0, 1))

    def test_keep_classes_filters(self):
        lines, *_ = parse_label(self.write("0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n"), {2})
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("0 0.100000"))

    def test_broken_coordinates_dropped(self):
        lines, dropped, *_ = parse_label(
            self.write("0 1.5 0.5 0.2 0.2\n0 0.5 0.5 0 0.2\n"))
        self.assertEqual(lines, [])
        self.assertEqual(dropped, 2)

    def test_multiclass_keeps_source_numbers(self):
        lines, _, counts, _ = parse_label(
            self.write("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"),
            collapse=False, num_classes=2)
        self.assertEqual([line.split()[0] for line in lines], ["0", "1"])
        self.assertEqual(dict(counts), {0: 1, 1: 1})

    def test_multiclass_mapping_applied(self):
        lines, _, counts, _ = parse_label(
            self.write("0 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1\n"),
            mapping={0: 1, 2: 0}, collapse=False, num_classes=2)
        self.assertEqual([line.split()[0] for line in lines], ["1", "0"])
        self.assertEqual(dict(counts), {0: 1, 1: 1})

    def test_classes_outside_map_are_dropped(self):
        lines, *_ = parse_label(self.write("5 0.5 0.5 0.2 0.2\n"),
                                mapping={0: 0}, collapse=False, num_classes=1)
        self.assertEqual(lines, [])

    def test_out_of_range_reported_not_silent(self):
        lines, _, _, out_of_range = parse_label(
            self.write("7 0.5 0.5 0.2 0.2\n"), collapse=False, num_classes=2)
        self.assertEqual(lines, [])
        self.assertEqual(out_of_range, {7})

    def test_split_is_deterministic(self):
        groups = [f"group-{index}" for index in range(400)]
        first = assign_splits(groups, 0.15, 0.1)
        self.assertEqual(first, assign_splits(list(reversed(groups)), 0.15, 0.1))

    def test_split_respects_ratios(self):
        groups = [f"group-{index}" for index in range(1000)]
        assignment = assign_splits(groups, 0.15, 0.1)
        share = lambda name: sum(1 for v in assignment.values() if v == name) / 1000
        self.assertAlmostEqual(share("val"), 0.15, places=2)
        self.assertAlmostEqual(share("test"), 0.10, places=2)

    def test_small_dataset_still_gets_every_split(self):
        assignment = assign_splits([f"g{i}" for i in range(7)], 0.15, 0.15)
        self.assertEqual(set(assignment.values()), {"train", "val", "test"})

    def test_single_group_goes_to_train(self):
        self.assertEqual(assign_splits(["one"], 0.15, 0.15), {"one": "train"})

    def test_group_key_modes(self):
        image = Path("/src/images/1-124_jpg.rf.02a45ed38379f3c0e35734b25ed5c4e1.jpg")
        self.assertEqual(group_key(image, Path("/src"), "roboflow"), "1-124_jpg")
        self.assertEqual(group_key(image, Path("/src"), "stem"), image.stem)
        self.assertEqual(
            group_key(Path("/src/TECH/CSQU3054383/images/a.jpg"), Path("/src"), "parent"),
            "TECH/CSQU3054383")
        self.assertEqual(
            group_key(Path("/src/images/CSQU3054383-2026.jpg"), Path("/src"),
                      r"regex:([A-Z]{4}[0-9]{7})"),
            "CSQU3054383")

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


class TestClassMap(unittest.TestCase):
    """Сопоставление классов источника классам проекта."""

    def setUp(self):
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.source = Path(self.temp.name) / "roboflow-a"
        self.source.mkdir()
        (self.source / "data.yaml").write_text(
            "names:\n  0: code\n  1: plate\n  2: junk\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_names_read_from_source(self):
        self.assertEqual(source_class_names(self.source),
                         {0: "code", 1: "plate", 2: "junk"})

    def test_names_from_list_form(self):
        (self.source / "data.yaml").write_text("names: [code, plate]\n", encoding="utf-8")
        self.assertEqual(source_class_names(self.source), {0: "code", 1: "plate"})

    def test_mapping_by_names(self):
        mapping = resolve_class_map({"code": "container_code", "plate": "plate"},
                                    self.source, source_class_names(self.source),
                                    ["container_code", "plate"])
        self.assertEqual(mapping, {0: 0, 1: 1})

    def test_mapping_by_numbers(self):
        mapping = resolve_class_map({0: 1, 2: 0}, self.source, {}, ["a", "b"])
        self.assertEqual(mapping, {0: 1, 2: 0})

    def test_per_source_mapping(self):
        raw = {"default": {0: 0}, "roboflow-a": {1: 0}}
        self.assertEqual(resolve_class_map(raw, self.source, {}, ["a"]), {1: 0})

    def test_unknown_source_class_is_an_error(self):
        with self.assertRaises(SystemExit):
            resolve_class_map({"seal": "plate"}, self.source,
                              source_class_names(self.source), ["plate"])

    def test_unknown_target_class_is_an_error(self):
        with self.assertRaises(SystemExit):
            resolve_class_map({"code": "seal"}, self.source,
                              source_class_names(self.source), ["plate"])


class TestStableSplits(unittest.TestCase):
    """«Липкие» сплиты: известные группы не переезжают при пополнении данных."""

    def test_known_groups_keep_their_split(self):
        groups = [f"g{i}" for i in range(40)]
        first = assign_splits(groups, 0.15, 0.15)
        grown = groups + [f"new{i}" for i in range(40)]
        second, fresh = extend_assignment(first, grown, 0.15, 0.15)
        self.assertEqual(fresh, 40)
        for key, split in first.items():
            self.assertEqual(second[key], split, f"группа {key} переехала")

    def test_new_groups_pull_ratios_to_target(self):
        first = assign_splits([f"g{i}" for i in range(20)], 0.15, 0.15)
        grown = list(first) + [f"new{i}" for i in range(180)]
        second, _ = extend_assignment(first, grown, 0.15, 0.15)
        share = lambda name: sum(1 for v in second.values() if v == name) / len(second)
        self.assertAlmostEqual(share("val"), 0.15, delta=0.02)
        self.assertAlmostEqual(share("test"), 0.15, delta=0.02)

    def test_extension_is_deterministic(self):
        first = assign_splits([f"g{i}" for i in range(10)], 0.2, 0.0)
        grown = list(first) + [f"new{i}" for i in range(10)]
        a, _ = extend_assignment(first, grown, 0.2, 0.0)
        b, _ = extend_assignment(first, list(reversed(grown)), 0.2, 0.0)
        self.assertEqual(a, b)

    def test_empty_history_falls_back_to_plain_split(self):
        groups = [f"g{i}" for i in range(30)]
        assignment, fresh = extend_assignment({}, groups, 0.15, 0.15)
        self.assertEqual(assignment, assign_splits(groups, 0.15, 0.15))
        self.assertEqual(fresh, 30)

    def test_signature_changes_with_assignment(self):
        base = assign_splits([f"g{i}" for i in range(20)], 0.15, 0.0)
        moved = dict(base)
        moved["g0"] = "val" if base["g0"] != "val" else "train"
        self.assertNotEqual(signature(base), signature(moved))
        self.assertEqual(signature(base), signature(dict(base)))


class TestNegatives(unittest.TestCase):
    """Ограничение доли кадров без объектов."""

    class _Stub:
        """Минимальный «проект»: функции нужен только доступ к настройке."""

        def __init__(self, limit):
            self.limit = limit

        def get(self, key, default=None):
            return self.limit if key == "dataset.negatives.max_share" else default

    def _fixture(self, positives: int, negatives: int):
        from mlkit.dataset.discover import Pair

        pairs, parsed, assignment = [], {}, {}
        for index in range(positives + negatives):
            pair = Pair(image=Path(f"/src/images/{index}.jpg"), label=Path(f"/src/{index}.txt"),
                        prefix="src", group=f"g{index}")
            pairs.append(pair)
            parsed[pair.key] = (["0 0.5 0.5 0.1 0.1"] if index < positives else [], 0)
            assignment[pair.group] = "train"
        return pairs, parsed, assignment

    def test_no_limit_keeps_everything(self):
        pairs, parsed, assignment = self._fixture(5, 50)
        skipped = _limit_negatives(self._Stub(None), pairs, parsed, assignment)
        self.assertEqual(skipped, set())

    def test_share_is_counted_from_final_size(self):
        pairs, parsed, assignment = self._fixture(90, 90)
        skipped = _limit_negatives(self._Stub(0.1), pairs, parsed, assignment)
        kept = len(pairs) - len(skipped)
        negatives_kept = 90 - len(skipped)
        self.assertEqual(negatives_kept / kept, 0.1)

    def test_positives_are_never_dropped(self):
        pairs, parsed, assignment = self._fixture(4, 40)
        skipped = _limit_negatives(self._Stub(0.0), pairs, parsed, assignment)
        self.assertEqual(len(skipped), 40)
        self.assertTrue(all(parsed[key][0] == [] for key in skipped))

    def test_selection_is_deterministic(self):
        pairs, parsed, assignment = self._fixture(10, 40)
        first = _limit_negatives(self._Stub(0.2), pairs, parsed, assignment)
        second = _limit_negatives(self._Stub(0.2), list(reversed(pairs)), parsed, assignment)
        self.assertEqual(first, second)

    def test_limit_applies_per_split(self):
        pairs, parsed, assignment = self._fixture(20, 20)
        for index, pair in enumerate(pairs):
            assignment[pair.group] = "train" if index % 2 else "val"
        skipped = _limit_negatives(self._Stub(0.25), pairs, parsed, assignment)
        for split in ("train", "val"):
            kept = [p for p in pairs if assignment[p.group] == split and p.key not in skipped]
            negatives = [p for p in kept if not parsed[p.key][0]]
            self.assertLessEqual(len(negatives) / len(kept), 0.25 + 1e-9)


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
