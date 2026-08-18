# Стадия 3 — сборка train/val из размеченных источников.
# Источники — все подкаталоги datasets/<P>/ (YOLO-формат).

.PHONY: dataset dataset-verify dataset-stats

dataset: check-project ## 1) собрать датасет: make dataset P=… [SOURCE=путь]
	$(MLKIT) dataset $(P) $(call opt,--source,$(SOURCE)) $(ARGS)

dataset-verify: check-project ## 1a) проверить источники, ничего не записывая
	$(MLKIT) dataset $(P) $(call opt,--source,$(SOURCE)) --verify $(ARGS)

dataset-stats: check-project ## 1b) статистика собранного датасета
	$(MLKIT) dataset-stats $(P)
