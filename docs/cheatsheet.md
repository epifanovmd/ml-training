# Шпаргалка

Все команды принимают проект через `P=<имя>`. Полный список — `make help`,
флаги конкретной стадии — `make cli ARGS="train --help"`.

## Команды по стадиям

### Окружение и состояние

| Команда | Что делает |
|---|---|
| `make install` | окружение сборки и обучения (`.venv`) |
| `make install-export` | окружение экспорта (`.venv-export`) |
| `make doctor P=…` | python, torch, устройство, источники данных |
| `make projects` | список проектов |
| `make status P=…` | что уже сделано: источники, датасет, прогоны, экспорты |
| `make test` | тесты ядра (без сети и весов) |

### 1. Датасет

| Команда | Что делает |
|---|---|
| `make dataset-verify P=…` | посчитать пары, группы, классы — ничего не записывая |
| `make dataset P=…` | собрать `train/val/test` + `data.yaml` + `manifest.json` |
| `make dataset P=… GROUP_BY=roboflow` | сменить единицу сплита разово |
| `make dataset P=… TEST_RATIO=0.15` | добавить отложенный сплит |
| `make dataset P=… ARGS="--max-negatives 0.1"` | ограничить долю фоновых кадров |
| `make dataset-reset-splits P=…` | забыть прежнюю раскладку и поделить заново |
| `make dataset-stats P=…` | боксы, классы, доля фона, проверка утечки |

### 2. Обучение

| Команда | Что делает |
|---|---|
| `make train-dry P=… PROFILE=quality` | итоговые параметры и предполётные проверки, без обучения |
| `make train P=…` | обучить по конфигу проекта |
| `make train P=… PROFILE=fast` | быстрый прогон «жив ли пайплайн» |
| `make train P=… PROFILE=quality` | выжать максимум (imgsz 960, 300 эпох, AdamW) |
| `make train P=… EPOCHS=3 DEVICE=cpu` | разовые переопределения |
| `make train P=… MODEL=yolo11s.pt` | другая базовая модель |
| `make train P=… ARGS="--set freeze=10 --set cache=ram"` | любой параметр ultralytics разово |
| `make finetune P=… FROM=<прогон>` | дообучить поверх своего прогона (профиль `finetune`) |
| `make resume P=… NAME=<прогон>` | продолжить прерванное обучение |
| `make runs P=…` | таблица прогонов, лучший сверху |
| `make val P=…` / `make test-metrics P=…` | метрики на val / на отложенном сплите |
| `make errors P=… LIMIT=30` | худшие кадры + `problems.csv` с путями в источниках |
| `make bench P=…` | мс/кадр, FPS, размер модели |
| `make predict P=… SHOW=1` | прогнать по своим картинкам из `samples/` |
| `make tune P=… ITERATIONS=20 EPOCHS=15` | генетический подбор гиперпараметров |

### 3. Экспорт

| Команда | Что делает |
|---|---|
| `make export P=… [RUN=<прогон>]` | CoreML + TFLite в `exports/{ios,android}` |
| `make export-check P=…` | сверить боксы экспорта с исходными весами |
| `make bench-export P=…` | скорость мобильных форматов |
| `make export-pretrained MODEL=yolo11n.pt` | экспорт готовой COCO-модели |
| `make pipeline P=…` | `dataset → train → export` одной командой |

### Уборка

| Команда | Что делает |
|---|---|
| `make clean P=…` | удалить собранный датасет и прогоны |
| `make clean-all P=…` | удалить всё генерируемое по проекту |

## Типичные сценарии

### Первый запуск на новом датасете

```bash
cp -r <датасет> datasets/<проект>/партия-1
make dataset-verify P=<проект>          # распознались ли источники и классы
make dataset        P=<проект>
make dataset-stats  P=<проект>          # доля фона, баланс классов, утечка
make train-dry      P=<проект>          # проверки до долгого прогона
make train          P=<проект> PROFILE=quality NAME=v1
make test-metrics   P=<проект> RUN=v1   # честная цифра
```

### Пришла новая порция данных

```bash
cp -r <новая порция> datasets/<проект>/партия-2
make dataset P=<проект>                 # старые группы останутся в своих сплитах
make train   P=<проект> PROFILE=quality NAME=v2
make runs    P=<проект>                 # сравнить v1 и v2
```

Полное переобучение — режим по умолчанию при заметном пополнении. Тёплый
старт (`FROM=`) экономит время, но при большом приросте данных даёт менее
предсказуемый результат.

### Дообучение поверх прошлого прогона

```bash
make finetune P=<проект> FROM=v1 NAME=v2-ft
make test-metrics P=<проект> RUN=v2-ft
make errors P=<проект> RUN=v2-ft
```

Учить всегда на **полном** датасете, а не только на новых кадрах: иначе
модель забывает то, что умела. Профиль `finetune` замораживает backbone и
снижает lr — это защита от разрушения признаков на малом пополнении.

### Модель ошибается — что дальше

```bash
make errors P=<проект> LIMIT=30
open workspace/<проект>/previews/errors-val/          # смотрим глазами
# правим разметку в datasets/… по problems.csv
make dataset P=<проект> && make train P=<проект> NAME=v3
make test-metrics P=<проект> RUN=v3                   # сравниваем на том же сплите
```

Зелёный бокс на превью — разметка, которую модель не нашла; розовый —
срабатывание, которого нет в разметке. Часто розовый оказывается прав.

### Подготовка к переносу в приложение

```bash
make runs         P=<проект>            # выбрать прогон
make export       P=<проект> RUN=v2
make export-check P=<проект> RUN=v2     # боксы экспорта совпадают с .pt?
make bench-export P=<проект>            # укладывается ли в бюджет по времени
```

### Эксперименты

```bash
make train P=<проект> MODEL=yolo11s.pt NAME=v2-s      # модель покрупнее
make train P=<проект> ARGS="--set imgsz=960" NAME=v2-960
make train P=<проект> ARGS="--set multi_scale=0.3" NAME=v2-ms
make runs  P=<проект>                                  # сравнить всё разом
```

Имя прогона задавайте всегда: без `NAME=` повторный запуск с теми же
параметрами перезапишет предыдущий (об этом предупредит).

## Переменные make

| Переменная | Где применяется |
|---|---|
| `P=` | проект (обязательно почти везде) |
| `PROFILE=` | профиль обучения: `fast`, `quality`, `finetune` |
| `FROM=` | прогон-источник весов для `make finetune` |
| `NAME=` | имя прогона |
| `RUN=` | чей `best.pt` брать для val/errors/bench/export |
| `EPOCHS= DEVICE= BATCH=` | разовые переопределения обучения |
| `MODEL=` | базовые веса (`yolo11s.pt`) или модель для `export-pretrained` |
| `GROUP_BY= TEST_RATIO=` | параметры сборки датасета |
| `INPUT= CONF= LIMIT= SHOW=` | проверка модели и разбор ошибок |
| `ARGS="…"` | любые флаги CLI напрямую |
