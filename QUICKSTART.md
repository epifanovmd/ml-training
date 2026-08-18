# Краткая инструкция

Обучение детектора «региона интереса»: на входе — размеченный датасет,
на выходе — модели для iOS и Android.

```
datasets/<проект>/  →  dataset  →  train  →  export
   размеченные        train/val    best.pt   .mlpackage / .tflite
     данные
```

## 1. Установка (один раз)

```bash
make install            # окружение обучения (.venv)
make install-export     # окружение экспорта (.venv-export)
make doctor P=container-code    # проверить, что всё готово
```

## 2. С чего начать

```bash
# 1) положить размеченные данные: пары images/*.jpg + labels/*.txt
cp -r <ваш-датасет> datasets/container-code/мой-датасет

# 2) собрать train/val
make dataset P=container-code

# 3) обучить (перед долгим прогоном полезно: make train-dry P=… PROFILE=quality)
make train P=container-code

# 4) посмотреть глазами на картинках из projects/<проект>/samples/
make predict P=container-code SHOW=1

# 5) получить модели для приложения
make export P=container-code
```

Готовые файлы окажутся в `workspace/container-code/exports/ios/` и
`.../android/` — их переносят в приложение вручную.

Всё сразу: `make pipeline P=container-code`.

## 3. Где что лежит

| Путь | Что там |
|---|---|
| `projects/<проект>/project.yaml` | **вся настройка задачи**: классы (один или несколько + `class_map`), базовая модель, эпохи, аугментации, профили обучения, имя экспортируемой модели |
| `projects/<проект>/samples/` | картинки для `make predict` |
| `datasets/<проект>/<имя>/` | размеченные датасеты; подкаталогов может быть сколько угодно, все попадут в сборку |
| `workspace/<проект>/dataset/` | собранный train/val/test + `data.yaml` и `manifest.json` (**пересоздаётся при каждой сборке**) |
| `workspace/<проект>/runs/<прогон>/` | обучение: `weights/best.pt`, графики, метрики, `run_info.json` и `dataset_manifest.json` — на чём и чем обучено |
| `workspace/<проект>/exports/` | **готовые модели**: `ios/*.mlpackage`, `android/*.tflite` |
| `weights/` | кэш базовых чекпойнтов (`yolo11n.pt` и т.п.) |
| `src/mlkit/` | ядро; при добавлении новой задачи его трогать не нужно |

Всё в `workspace/` — генерируемое: руками не правится, восстанавливается
командами. Правки вносятся в источники (`datasets/`) и в `project.yaml`.

## 4. Шпаргалка

```bash
make help                       # все команды
make projects                   # список задач
make status P=container-code    # что уже сделано по задаче

make dataset-verify P=…               # сколько пар нашлось, ничего не записывая
make dataset-stats  P=…               # боксы, негативные, утечка между сплитами
make dataset P=… GROUP_BY=roboflow    # сплит по группам, а не по кадрам
make dataset P=… ARGS="--max-negatives 0.1"   # не больше 10% кадров без объектов

make train-dry P=… PROFILE=quality    # итоговые параметры и проверки, без обучения
make train P=… PROFILE=fast           # быстрая проверка идеи (15 эпох)
make train P=… PROFILE=quality        # выжать максимум (imgsz 960, 300 эпох, AdamW)
make train P=… EPOCHS=3 DEVICE=cpu    # разовые переопределения
make train P=… ARGS="--set freeze=10" # любой параметр ultralytics разово
make resume P=…                       # продолжить прерванное обучение

make runs P=…                         # таблица прогонов, лучший сверху
make val P=… ; make test-metrics P=…  # метрики на val / на отложенном сплите
make errors P=… LIMIT=30              # кадры, где модель ошибается
make bench P=…                        # мс/кадр, FPS, размер модели

make export P=… RUN=<прогон>          # экспорт конкретного прогона
make export-check P=…                 # сверить экспорт с исходными весами
make clean P=…                        # удалить датасет и прогоны задачи

make cli ARGS="train --help"          # все флаги любой стадии
```

## 5. Новая задача (например, детектор пломб)

```bash
mkdir -p projects/seal/samples
cp projects/plate/project.yaml projects/seal/project.yaml
$EDITOR projects/seal/project.yaml    # description, classes, export.model_name
```

Дальше положить размеченные данные в `datasets/seal/` и запустить
`make dataset P=seal && make train P=seal`. Код при этом не меняется.

## 6. Что читать дальше

- `README.md` — подробно по каждой стадии и все правила
- `docs/architecture.md` — как устроено внутри и как расширять
