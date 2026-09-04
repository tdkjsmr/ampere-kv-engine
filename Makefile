PYTHON ?= python3
RUN_ID ?= dev
RESULT_DIR ?= results/runs/$(RUN_ID)

.PHONY: help bootstrap build verify-env test-cpu test-gpu smoke-cuda smoke-triton \
	model-contract capture-hf-reference release-check

help:
	@echo "AmpereKV v0.1 可用命令："
	@echo "  make bootstrap             创建/更新开发环境"
	@echo "  make build                 为 sm_86 构建 CUDA Extension"
	@echo "  make verify-env            生成环境指纹 JSON"
	@echo "  make test-cpu              运行不依赖 GPU 的测试"
	@echo "  make test-gpu              运行 CUDA/Triton 测试"
	@echo "  make smoke-cuda            运行最小 CUDA Extension Smoke"
	@echo "  make smoke-triton          运行最小 Triton Kernel Smoke"
	@echo "  make model-contract        校验 Qwen3-8B 模型契约"
	@echo "  make capture-hf-reference  生成固定 Prompt 的 HF 基线"

bootstrap:
	bash scripts/bootstrap.sh

build:
	AMPERE_KV_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 \
		$(PYTHON) -m pip install -v -e . --no-build-isolation

verify-env:
	bash scripts/verify_environment.sh --output "$(RESULT_DIR)/environment.json"

test-cpu:
	$(PYTHON) -m pytest -m "not gpu" tests/cpu

test-gpu:
	$(PYTHON) -m pytest -m gpu tests/gpu

smoke-cuda:
	$(PYTHON) scripts/run_cuda_smoke.py \
		--output "$(RESULT_DIR)/cuda_smoke.json"

smoke-triton:
	$(PYTHON) scripts/run_triton_smoke.py \
		--output "$(RESULT_DIR)/triton_smoke.json"

model-contract:
	$(PYTHON) -m ampere_kv.model.contract \
		--lock configs/model.lock.yaml \
		--config tests/fixtures/qwen3_8b_config.json

capture-hf-reference:
	$(PYTHON) scripts/capture_hf_reference.py \
		--lock configs/model.lock.yaml \
		--prompts local_private/reference_prompts.json \
		--output "$(RESULT_DIR)/hf_reference.json"

release-check:
	$(PYTHON) scripts/release_check.py --run-dir "$(RESULT_DIR)"
