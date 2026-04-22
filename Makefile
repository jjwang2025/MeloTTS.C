BUILD_DIR := build
MODEL_DIR := models
THIRD_PARTY_DIR := third_party
ONNXRUNTIME_VERSION := 1.20.1
ONNXRUNTIME_PACKAGE := onnxruntime-win-x64-$(ONNXRUNTIME_VERSION)
ONNXRUNTIME_ROOT ?= $(CURDIR)/$(THIRD_PARTY_DIR)/$(ONNXRUNTIME_PACKAGE)
ONNXRUNTIME_ZIP := $(THIRD_PARTY_DIR)/$(ONNXRUNTIME_PACKAGE).zip
ONNXRUNTIME_URL := https://github.com/microsoft/onnxruntime/releases/download/v$(ONNXRUNTIME_VERSION)/$(ONNXRUNTIME_PACKAGE).zip

.PHONY: all deps model ensure-models ensure-bert ensure-acoustic clean distclean purge

all: deps ensure-models
	cmake -S . -B $(BUILD_DIR) -DONNXRUNTIME_ROOT="$(ONNXRUNTIME_ROOT)"
	cmake --build $(BUILD_DIR) --config Release

deps:
	powershell -NoProfile -Command "if (!(Test-Path '$(THIRD_PARTY_DIR)')) { New-Item -ItemType Directory -Path '$(THIRD_PARTY_DIR)' | Out-Null }"
	powershell -NoProfile -Command "if (!(Test-Path '$(ONNXRUNTIME_ROOT)')) { Invoke-WebRequest -Uri '$(ONNXRUNTIME_URL)' -OutFile '$(ONNXRUNTIME_ZIP)' }"
	powershell -NoProfile -Command "if (!(Test-Path '$(ONNXRUNTIME_ROOT)')) { Expand-Archive -Path '$(ONNXRUNTIME_ZIP)' -DestinationPath '$(THIRD_PARTY_DIR)' -Force }"

model: deps ensure-models

ensure-models:
	@$(MAKE) ensure-bert
	@$(MAKE) ensure-acoustic

ensure-bert: deps
	powershell -NoProfile -Command "if (!(Test-Path '$(MODEL_DIR)/bert_base_uncased.onnx') -or !(Test-Path '$(MODEL_DIR)/bert-base-uncased-vocab.txt')) { python tools/export_english_onnx.py --output-dir $(MODEL_DIR) --skip-acoustic }"

ensure-acoustic: deps
	powershell -NoProfile -Command "if (!(Test-Path '$(MODEL_DIR)/melotts_en_infer.onnx')) { python tools/export_english_onnx.py --output-dir $(MODEL_DIR) --skip-bert }"

clean:
	powershell -NoProfile -Command "if (Test-Path '$(BUILD_DIR)') { Remove-Item '$(BUILD_DIR)' -Recurse -Force }"
	powershell -NoProfile -Command "if (Test-Path 'outputs') { Remove-Item 'outputs' -Recurse -Force }"
	powershell -NoProfile -Command "if (Test-Path 'tools/__pycache__') { Remove-Item 'tools/__pycache__' -Recurse -Force }"
	powershell -NoProfile -Command "if (Test-Path 'tools/melotts_py/__pycache__') { Remove-Item 'tools/melotts_py/__pycache__' -Recurse -Force }"
	powershell -NoProfile -Command "if (Test-Path 'tools/melotts_py/monotonic_align/__pycache__') { Remove-Item 'tools/melotts_py/monotonic_align/__pycache__' -Recurse -Force }"
	powershell -NoProfile -Command "if (Test-Path 'CMakeFiles') { Remove-Item 'CMakeFiles' -Recurse -Force }"
	powershell -NoProfile -Command "if (Test-Path 'CMakeCache.txt') { Remove-Item 'CMakeCache.txt' -Force }"
	powershell -NoProfile -Command "if (Test-Path 'cmake_install.cmake') { Remove-Item 'cmake_install.cmake' -Force }"

distclean: clean
	powershell -NoProfile -Command "if (Test-Path 'third_party/onnxruntime-win-x64-1.20.1') { Remove-Item 'third_party/onnxruntime-win-x64-1.20.1' -Recurse -Force }"

purge: distclean
	powershell -NoProfile -Command "if (Test-Path '$(MODEL_DIR)') { Remove-Item '$(MODEL_DIR)' -Recurse -Force }"
