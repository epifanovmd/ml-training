# Зависимости по ролям

| Файл | Стадии | Окружение |
|---|---|---|
| `base.txt` | ядро (конфиги, изображения) | `.venv` |
| `train.txt` | 1–2 — датасет и обучение | `.venv` |
| `export.txt` | 3 — CoreML / TFLite | `.venv-export` |

Версии в `train.txt` закреплены точно: обновление ultralytics меняет и
обучение, и формат экспорта, а пара torch/torchvision должна совпадать по
минорной версии. Автоустановка пакетов ultralytics отключена в коде
(`YOLO_AUTOINSTALL=false`) — иначе он доустанавливает зависимости прямо в
рабочее окружение по ходу команды.

Окружения намеренно разделены: в обучающем `.venv` экспорт TFLite ломает
связку torch/torchvision (ultralytics 8.4 тянет litert-torch и даунгрейдит
torch), а coremltools 9 несовместим с numpy 2.5. В `.venv-export`
закреплена проверенная связка ultralytics 8.3 + coremltools 8 + onnx2tf.
