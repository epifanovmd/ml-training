# Зависимости по ролям

| Файл | Стадии | Окружение |
|---|---|---|
| `base.txt` | ядро (конфиги, изображения) | `.venv` |
| `train.txt` | 1–2 — датасет и обучение | `.venv` |
| `export.txt` | 3 — CoreML / TFLite | `.venv-export` |

Окружения намеренно разделены: в обучающем `.venv` экспорт TFLite ломает
связку torch/torchvision (ultralytics 8.4 тянет litert-torch и даунгрейдит
torch), а coremltools 9 несовместим с numpy 2.5. В `.venv-export`
закреплена проверенная связка ultralytics 8.3 + coremltools 8 + onnx2tf.
