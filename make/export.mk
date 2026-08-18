# Стадия 5 — конвертация в мобильные форматы. Отдельное окружение .venv-export.

.PHONY: export export-check export-pretrained bench-export pipeline check-export-venv

check-export-venv:
	@test -x $(PY_EXPORT) || { echo "Нет $(VENV_EXPORT) — сначала: make install-export"; exit 1; }

export: check-project check-export-venv ## 3) экспорт в CoreML/TFLite: make export P=… [RUN=]
	$(MLKIT_EXPORT) export $(P) $(call opt,--run,$(RUN)) $(call opt,--format,$(FORMAT)) $(ARGS)

# Форматы проверяются в разных окружениях: CoreML читается coremltools 9
# из .venv, а TFLite — только рантаймом из .venv-export. Запуск CoreML
# в .venv-export падает по assert внутри MPSGraph, поэтому не смешиваем.
export-check: check-project check-export-venv ## 3a) сверить боксы экспорта с весами .pt
	$(MLKIT) export-check $(P) --format coreml $(call opt,--run,$(RUN)) $(ARGS)
	$(MLKIT_EXPORT) export-check $(P) --format tflite $(call opt,--run,$(RUN)) $(ARGS)

bench-export: check-project check-export-venv ## 3b) скорость мобильных форматов
	$(MLKIT) bench $(P) --only mlpackage $(call opt,--run,$(RUN)) $(ARGS)
	$(MLKIT_EXPORT) bench $(P) --only tflite $(call opt,--run,$(RUN)) $(ARGS)

export-pretrained: check-export-venv ## 3c) экспорт готовой модели: make export-pretrained MODEL=yolo11n.pt
	$(MLKIT_EXPORT) export --pretrained $(if $(MODEL),$(MODEL),yolo11n.pt) $(ARGS)

pipeline: check-project ## прогнать стадии подряд: make pipeline P=… [STAGES=dataset,train,export]
	$(MLKIT) pipeline $(P) $(call opt,--stages,$(STAGES)) $(call opt,--epochs,$(EPOCHS)) \
		$(call opt,--device,$(DEVICE)) $(call opt,--name,$(NAME)) $(ARGS)
