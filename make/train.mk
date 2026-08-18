# Стадия 4 — обучение и проверка модели.

.PHONY: train resume val test-metrics errors bench predict tune runs

train: check-project ## 2) обучить: make train P=… [PROFILE= EPOCHS= DEVICE= MODEL= NAME=]
	$(MLKIT) train $(P) $(call opt,--profile,$(PROFILE)) $(call opt,--epochs,$(EPOCHS)) \
		$(call opt,--device,$(DEVICE)) $(call opt,--batch,$(BATCH)) \
		$(call opt,--model,$(MODEL)) $(call opt,--name,$(NAME)) $(ARGS)

train-dry: check-project ## 2a) показать итоговые параметры и проверки, не обучая
	$(MLKIT) train $(P) $(call opt,--profile,$(PROFILE)) $(call opt,--epochs,$(EPOCHS)) \
		$(call opt,--device,$(DEVICE)) --dry-run $(ARGS)

resume: check-project ## 2b) продолжить прерванное обучение
	$(MLKIT) train $(P) --resume $(call opt,--name,$(NAME)) $(ARGS)

val: check-project ## 2c) метрики обученной модели на val
	$(MLKIT) val $(P) $(call opt,--run,$(RUN)) $(call opt,--device,$(DEVICE)) $(ARGS)

test-metrics: check-project ## 2d) честные метрики на отложенном test-сплите
	$(MLKIT) test-metrics $(P) $(call opt,--run,$(RUN)) $(call opt,--device,$(DEVICE)) $(ARGS)

errors: check-project ## 2e) кадры, где модель ошибается: make errors P=… [LIMIT= CONF=]
	$(MLKIT) errors $(P) $(call opt,--run,$(RUN)) $(call opt,--conf,$(CONF)) \
		$(call opt,--limit,$(LIMIT)) $(call opt,--device,$(DEVICE)) $(ARGS)

bench: check-project ## 2f) скорость весов и экспортов (мс/кадр, FPS, размер)
	$(MLKIT) bench $(P) $(call opt,--run,$(RUN)) $(call opt,--device,$(DEVICE)) $(ARGS)

predict: check-project ## 2g) прогнать модель по картинкам: make predict P=… [INPUT= CONF=]
	$(MLKIT) predict $(P) $(call optq,--input,$(INPUT)) $(call opt,--conf,$(CONF)) \
		$(call opt,--run,$(RUN)) $(call opt,--device,$(DEVICE)) $(if $(SHOW),--show) $(ARGS)

tune: check-project ## 2h) подобрать гиперпараметры: make tune P=… [ITERATIONS= EPOCHS=]
	$(MLKIT) tune $(P) $(call opt,--iterations,$(ITERATIONS)) $(call opt,--epochs,$(EPOCHS)) \
		$(call opt,--profile,$(PROFILE)) $(call opt,--device,$(DEVICE)) $(ARGS)

runs: check-project ## 2i) таблица прогонов с метриками
	$(MLKIT) runs $(P)
