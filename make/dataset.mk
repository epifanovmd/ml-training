# Стадия 3 — сборка train/val из размеченных источников.
# Источники — все подкаталоги datasets/<P>/ (YOLO-формат).

.PHONY: dataset dataset-verify dataset-stats dataset-reset-splits

dataset: check-project ## 1) собрать датасет: make dataset P=… [SOURCE= GROUP_BY= TEST_RATIO=]
	$(MLKIT) dataset $(P) $(call optq,--source,$(SOURCE)) $(call optq,--group-by,$(GROUP_BY)) \
		$(call opt,--test-ratio,$(TEST_RATIO)) $(ARGS)

dataset-reset-splits: check-project ## 1c) пересчитать train/val/test с нуля
	$(MLKIT) dataset $(P) --reset-splits $(ARGS)

dataset-verify: check-project ## 1a) проверить источники, ничего не записывая
	$(MLKIT) dataset $(P) $(call optq,--source,$(SOURCE)) --verify $(ARGS)

dataset-stats: check-project ## 1d) статистика собранного датасета
	$(MLKIT) dataset-stats $(P)
