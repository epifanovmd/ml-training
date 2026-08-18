# Общие переменные и проверки для всех стадий.

VENV        := .venv
PY          := $(VENV)/bin/python
VENV_EXPORT := .venv-export
PY_EXPORT   := $(VENV_EXPORT)/bin/python
# tensorflow (нужен для TFLite) не имеет колёс под Python 3.14+
PYVER       := 3.12

MLKIT        = PYTHONPATH=src $(PY) -m mlkit
MLKIT_EXPORT = PYTHONPATH=src $(PY_EXPORT) -m mlkit
MLKIT_TEST   = PYTHONPATH=src $(PY)

PROJECTS := $(notdir $(patsubst %/project.yaml,%,$(wildcard projects/*/project.yaml)))

# P — проект. TASK поддерживается для совместимости со старыми командами.
TASK ?=
P    ?= $(TASK)

# Часто используемые переопределения
MODEL      ?=
INPUT      ?=
GROUP_BY   ?=
TEST_RATIO ?=
PROFILE    ?=
ITERATIONS ?=
LIMIT      ?=
FORMAT     ?=
EPOCHS  ?=
DEVICE  ?=
BATCH   ?=
NAME    ?=
RUN     ?=
CONF    ?=
SOURCE  ?=
STAGES  ?=
SHOW    ?=
ARGS    ?=

opt = $(if $(2),$(1) $(2))
# То же, но значение в кавычках: для путей с пробелами и шаблонов вроде
# GROUP_BY='regex:([A-Z]{4}[0-9]{7})', которые иначе съест шелл
optq = $(if $(2),$(1) '$(2)')

.PHONY: check-project check-venv cli

check-venv:
	@test -x $(PY) || { echo "Нет $(VENV) — сначала: make install"; exit 1; }

check-project: check-venv
	@test -n "$(P)" || { echo "Нужен проект: make $(MAKECMDGOALS) P=<проект>. Доступные: $(PROJECTS)"; exit 1; }
	@test -f "projects/$(P)/project.yaml" || { echo "Нет projects/$(P)/project.yaml. Доступные: $(PROJECTS)"; exit 1; }

cli: check-venv ## произвольная команда CLI: make cli ARGS="status container-code"
	$(MLKIT) $(ARGS)
