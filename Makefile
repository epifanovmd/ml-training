# ml-training — обучение детекторов из готовых размеченных датасетов.
#
#   make help                        список команд
#   make install                     окружение обучения
#   make install-export              окружение экспорта моделей
#   make pipeline P=container-code   dataset -> train -> export
#
# Размеченные данные кладутся в datasets/<проект>/.
#
# Стадии разнесены по файлам в make/: install, dataset, train, export.
# Все они дергают один CLI: `.venv/bin/python -m mlkit`.

include make/common.mk
include make/install.mk
include make/dataset.mk
include make/train.mk
include make/export.mk

.DEFAULT_GOAL := help

help: ## показать этот список
	@echo "Конвейер: dataset -> train -> export"
	@echo "Данные: размеченные датасеты в datasets/<проект>/"
	@echo ""
	@echo "Команды:"
	@grep -hE '^[a-z][a-z-]*:.*## ' Makefile make/install.mk make/dataset.mk \
		make/train.mk make/export.mk make/common.mk \
		| sed 's/:.*## /\t/' | awk -F '\t' '{printf "  make %-16s %s\n", $$1, $$2}'
	@echo ""
	@echo "Переменные:"
	@echo "  P=<проект>        проект из projects/ (доступные: $(PROJECTS))"
	@echo "  MODEL=yolo11s.pt  базовые веса вместо model.base"
	@echo "  EPOCHS= DEVICE=   переопределить обучение (mps, cpu, 0)"
	@echo "  INPUT=<папка>     картинки для predict (по умолчанию samples/)"
	@echo "  RUN=<прогон>      чьи веса брать для predict/export"
	@echo "  ARGS=\"…\"          любые дополнительные флаги CLI"
	@echo ""
	@echo "Полный список флагов стадии: make cli ARGS=\"<команда> --help\""

.PHONY: help
