# Установка окружений. Используется uv (сам скачает нужный Python),
# при его отсутствии — venv системного python3.12.

.PHONY: install install-export doctor projects status test clean clean-all

define venv_install
	@if command -v uv >/dev/null 2>&1; then \
		{ test -x $(1)/bin/python || uv venv --python $(PYVER) --seed $(1); } && \
		uv pip install --python $(1)/bin/python -r $(2); \
	else \
		{ test -x $(1)/bin/python || python$(PYVER) -m venv $(1); } && \
		$(1)/bin/pip install -r $(2); \
	fi
endef

install: ## создать .venv (сборка датасета и обучение)
	$(call venv_install,$(VENV),requirements/train.txt)

install-export: ## создать .venv-export (экспорт: CoreML / TFLite)
	$(call venv_install,$(VENV_EXPORT),requirements/export.txt)

doctor: check-venv ## проверить окружение и данные проекта: make doctor P=…
	$(MLKIT) doctor $(P) $(ARGS)

projects: check-venv ## список проектов
	$(MLKIT) projects

status: check-project ## состояние всех стадий проекта
	$(MLKIT) status $(P)

clean: check-project ## удалить собранный датасет и прогоны проекта
	rm -rf workspace/$(P)/dataset workspace/$(P)/runs workspace/$(P)/previews
	@echo "Осталось: workspace/$(P)/exports и сами источники в datasets/$(P)/"

clean-all: check-project ## удалить всё генерируемое по проекту (включая экспорты)
	rm -rf workspace/$(P)

test: check-venv ## прогнать тесты ядра (без сети и весов)
	$(MLKIT_TEST) -m unittest discover -s tests -v
