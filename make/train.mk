# Стадия 4 — обучение и проверка модели.

.PHONY: train resume val predict

train: check-project ## 2) обучить: make train P=… [EPOCHS= DEVICE= MODEL= NAME=]
	$(MLKIT) train $(P) $(call opt,--epochs,$(EPOCHS)) $(call opt,--device,$(DEVICE)) \
		$(call opt,--batch,$(BATCH)) $(call opt,--model,$(MODEL)) \
		$(call opt,--name,$(NAME)) $(ARGS)

resume: check-project ## 2a) продолжить прерванное обучение
	$(MLKIT) train $(P) --resume $(call opt,--name,$(NAME)) $(ARGS)

val: check-project ## 2b) метрики обученной модели на val
	$(MLKIT) val $(P) $(call opt,--run,$(RUN)) $(call opt,--device,$(DEVICE)) $(ARGS)

predict: check-project ## 2c) прогнать модель по картинкам: make predict P=… [INPUT= CONF=]
	$(MLKIT) predict $(P) $(call opt,--input,$(INPUT)) $(call opt,--conf,$(CONF)) \
		$(call opt,--run,$(RUN)) $(call opt,--device,$(DEVICE)) $(if $(SHOW),--show) $(ARGS)
