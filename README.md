# ml-training

> Коротко и по делу — `QUICKSTART.md`. Ниже — подробности по каждой стадии.

Конвейер обучения детекторов «регионов интереса» для мобильного OCR.
Детектор находит область (код контейнера ISO 6346, автономер, что угодно),
текст читают системные движки приложения (Apple Vision / ML Kit) — их
не обучаем. На выходе — `.mlpackage` (iOS) и `.tflite` (Android).

## Три стадии

```
 datasets/<проект>/      1. dataset       2. train        3. export
 размеченные данные ──▶  train/val   ──▶  веса .pt   ──▶  .mlpackage
                         + data.yaml                       .tflite
```

На вход нужен размеченный датасет в YOLO-формате: пары `images/*.jpg` +
`labels/*.txt`. Откуда он взялся — ручная разметка, экспорт из Roboflow,
автоматический сборщик — конвейеру безразлично.

## Быстрый старт

```bash
make install                          # окружение обучения
make install-export                   # окружение экспорта

make projects                         # какие проекты есть
make status P=container-code          # что уже сделано

make dataset P=container-code         # 1) собрать train/val
make train   P=container-code         # 2) обучить
make export  P=container-code         # 3) .mlpackage + .tflite
```

Датасет — это каталог с парами `images/*.jpg` + `labels/*.txt` в
`datasets/<проект>/<любое-имя>/`; `make dataset` подхватывает все такие
каталоги сразу.

## Раскладка репозитория

| Каталог | Роль | В git |
|---|---|---|
| `src/mlkit/` | **ядро**: сборка датасета, обучение, экспорт. О контейнерах не знает | да |
| `projects/<имя>/` | **проект**: `project.yaml` и `samples/` для проверки | да |
| `make/` | по файлу на стадию, все дергают один CLI | да |
| `requirements/` | зависимости по ролям (base / train / export) | да |
| `datasets/<проект>/` | размеченные датасеты (кладутся сюда) | нет |
| `workspace/<проект>/` | всё генерируемое: `dataset/`, `runs/`, `exports/`, `previews/` | нет |
| `weights/` | кэш базовых чекпойнтов (`yolo11n.pt` и т.п.) | нет |

Ядро не содержит ни одной доменной строки: имена классов, гиперпараметры и
имя экспортируемой модели приходят из `project.yaml`. Поэтому новый
детектор — это новый файл конфигурации, а не правка ядра.

## Стадия 1 — датасет (`dataset`)

Объединяет источники, приводит классы, делает детерминированный сплит и
пишет `data.yaml` для ultralytics.

```bash
make dataset-verify P=container-code    # сколько пар нашлось, ничего не пишем
make dataset        P=container-code
make dataset-stats  P=container-code    # боксы, негативные, размеры объектов
```

Источники по умолчанию — все подкаталоги `datasets/<проект>/`; явный
список — в `dataset.sources`. Правила:

* `workspace/<проект>/dataset/` **полностью пересоздаётся** при каждой сборке —
  правки вносятся в источники, не в собранный датасет;
* все классы источника ремапятся в класс `0` (задача одноклассовая);
  нужны не все — `dataset.keep_classes: [0, 2]`;
* сплит детерминированный (sha1 от имени с префиксом источника): повторная
  сборка даёт тот же train/val, перенос источника в другой каталог — тоже,
  префикс считается от имени каталога, а не от полного пути;
* кадры без боксов попадают в датасет негативными — они снижают ложные
  срабатывания, выбрасывать их не нужно.

## Стадия 2 — обучение (`train`)

```bash
make train   P=container-code                       # по конфигу
make train   P=container-code EPOCHS=3 DEVICE=cpu   # быстрая проверка
make train   P=container-code MODEL=yolo11s.pt      # другая база -> runs/<проект>-yolo11s
make resume  P=container-code                       # после прерывания
make val     P=container-code                       # метрики на val
make predict P=container-code SHOW=1                # посмотреть глазами
```

Гиперпараметры и аугментации — только в `project.yaml`; флаги CLI нужны для
экспериментов. Прогоны лежат в `workspace/<проект>/runs/<прогон>/`,
метрики печатаются в конце. Ориентир пригодности: mAP50 ≥ 0.85.

## Стадия 3 — экспорт (`export`)

```bash
make export P=container-code                  # веса runs/<проект>/weights/best.pt
make export P=container-code RUN=container-code-yolo11s
make export-pretrained MODEL=yolo11n.pt       # готовая COCO-модель
```

Результат: `workspace/<проект>/exports/ios/<model_name>.mlpackage` и
`.../android/<model_name>.tflite`. Форматы задаются `export.formats`
(`coreml`, `tflite`, `onnx`, `torchscript`).

* CoreML по умолчанию **без встроенного NMS** — постобработку делает
  приложение; `export.coreml_nms: true` вернёт готовые боксы Apple Vision.
* `export.tflite_int8: true` — int8-квантизация (нужен собранный датасет
  для калибровки).
* Архитектура (классическая v8/11/12 или end-to-end v10/26) определяется по
  самим весам, так что экспорт корректен и для прогонов с `MODEL=`.

Экспорт идёт в отдельном окружении `.venv-export` — `make export` следит
за этим сам, `make pipeline` перезапускает в нём только стадию экспорта.

## Новый проект

```bash
mkdir -p projects/seal-number/samples
cp projects/plate/project.yaml projects/seal-number/project.yaml
$EDITOR projects/seal-number/project.yaml     # description, classes, export.model_name
```

Дальше кладём размеченный датасет в `datasets/seal-number/<имя>/`.
Ядро трогать не нужно.

## Окружения

| Окружение | Ставится | Для чего |
|---|---|---|
| `.venv` | `make install` | сборка датасета и обучение |
| `.venv-export` | `make install-export` | экспорт моделей |

Python 3.10–3.13 (**не 3.14+**: под него нет колёс tensorflow), в Makefile
закреплён 3.12; используется `uv`, при его отсутствии — `python3.12 -m venv`.
Окружения не смешивать: в обучающем `.venv` TFLite-экспорт ломает связку
torch/torchvision, а coremltools 9 несовместим с numpy 2.5.

```bash
make doctor P=container-code     # python, torch, mps, .venv-export, источники данных
```

## Частые сценарии

| Задача | Команды |
|---|---|
| Есть готовый датасет | положить в `datasets/<проект>/`, затем `make pipeline P=…` |
| Пришли новые данные | `make dataset P=… && make train P=…` |
| Полный цикл одной командой | `make pipeline P=…` |
| Проверить, как VLM видит одно фото | `make preview P=… IMAGE=<файл> SHOW=1` |
| Сравнить n и s | `make train P=…` и `make train P=… MODEL=yolo11s.pt`, сравнить mAP |
| Быстрая проверка пайплайна | `make train P=… EPOCHS=3` |
| Посмотреть модель глазами | картинки в `projects/<проект>/samples/`, затем `make predict P=…` |
| Начать задачу заново | `make clean P=…` (датасет и прогоны) или `make clean-all P=…` (всё) |

Полный список: `make help`; флаги любой стадии: `make cli ARGS="train --help"`.
Прямой запуск без make: `bin/mlkit status container-code`.
