# Simple Makefile shortcuts for BASE (2D) training & testing with uv
# You can rely entirely on train.py defaults. Provide overrides only when needed, e.g.:
#   make train EPOCHS=10 LR=1e-4
#   make train FAST_DEV=1
#   make test               (auto-picks latest checkpoint)
#   make test CKPT=path/to/checkpoint.ckpt

# ----- Core configuration (override as needed) -----
UV ?= uv
TRAIN_SCRIPT ?= train.py
EPOCHS ?=               # Pass to override --epochs
LR ?=                   # Pass to override --lr
BASE_CHANNELS ?=        # Pass to override --base-channels
BATCH_SIZE ?=           # Pass to override --batch-size
VAL_SPLIT ?=            # Pass to override --val-split
NUM_WORKERS ?=          # Pass to override --num-workers
SEED ?=                 # Pass to override --seed
FAST_DEV ?= 0           # Set to 1 to enable --fast-dev-run
DEBUG_RECON ?= 0        # Set to 1 to enable --debug-recon
CKPT ?=                 # Explicit checkpoint path for test target
EXTRA ?=                # Any extra passthrough args, e.g. EXTRA="--some-flag value"

# Experiment tag must match the one hard-coded in train.py
EXPERIMENT_TAG ?= mbg_train_synth_test_cal_exp_25bkgs

# Detect latest checkpoint (only evaluated in the shell within test target)
LATEST_CKPT = $(shell ls -1t lightning_logs/$(EXPERIMENT_TAG)/version_*/checkpoints/*.ckpt 2>/dev/null | head -n 1)

# Internal helper to conditionally add fast-dev-run flag
FAST_DEV_FLAG = $(if $(filter 1,$(FAST_DEV)),--fast-dev-run,)
DEBUG_RECON_FLAG = $(if $(filter 1,$(DEBUG_RECON)),--debug-recon,)

# Compose optional args only if user provided values
ifdef EPOCHS
	EPOCHS_ARG = --epochs $(EPOCHS)
endif
ifdef LR
	LR_ARG = --lr $(LR)
endif
ifdef BASE_CHANNELS
	BASE_CHANNELS_ARG = --base-channels $(BASE_CHANNELS)
endif
ifdef BATCH_SIZE
	BATCH_SIZE_ARG = --batch-size $(BATCH_SIZE)
endif
ifdef VAL_SPLIT
	VAL_SPLIT_ARG = --val-split $(VAL_SPLIT)
endif
ifdef NUM_WORKERS
	NUM_WORKERS_ARG = --num-workers $(NUM_WORKERS)
endif
ifdef SEED
	SEED_ARG = --seed $(SEED)
endif

.PHONY: help train test vars

help:
	@echo "Available targets:"
	@echo "  make train            - Run training (uses defaults unless overrides provided)"
	@echo "  make test             - Test-only run; loads latest or specified checkpoint"
	@echo "  make vars             - Print resolved variable values"
	@echo "\nCommon overrides: EPOCHS LR BASE_CHANNELS BATCH_SIZE VAL_SPLIT NUM_WORKERS SEED FAST_DEV DEBUG_RECON CKPT EXTRA"
	@echo "Example: make train EPOCHS=20 LR=5e-4 BATCH_SIZE=16"
	@echo "Example: make train FAST_DEV=1"
	@echo "Example: make test CKPT=path/to/file.ckpt"

vars:
	@echo "TRAIN_SCRIPT=$(TRAIN_SCRIPT)"
	@echo "EPOCHS=$(EPOCHS)"
	@echo "LR=$(LR)"
	@echo "BASE_CHANNELS=$(BASE_CHANNELS)"
	@echo "BATCH_SIZE=$(BATCH_SIZE)"
	@echo "VAL_SPLIT=$(VAL_SPLIT)"
	@echo "NUM_WORKERS=$(NUM_WORKERS)"
	@echo "SEED=$(SEED)"
	@echo "FAST_DEV=$(FAST_DEV)"
	@echo "DEBUG_RECON=$(DEBUG_RECON)"
	@echo "CKPT=$(CKPT)"
	@echo "EXTRA=$(EXTRA)"
	@echo "EXPERIMENT_TAG=$(EXPERIMENT_TAG)"

train:
	@echo "[TRAIN] Script: $(TRAIN_SCRIPT) (defaults unless overridden)"
	$(UV) run $(TRAIN_SCRIPT) \
	  $(EPOCHS_ARG) \
	  $(LR_ARG) \
	  $(BASE_CHANNELS_ARG) \
	  $(BATCH_SIZE_ARG) \
	  $(VAL_SPLIT_ARG) \
	  $(NUM_WORKERS_ARG) \
	  $(SEED_ARG) \
	  $(FAST_DEV_FLAG) \
	  $(DEBUG_RECON_FLAG) \
	  $(EXTRA)

# Test-only run: load explicit CKPT if provided, else fall back to latest discovered.
# Fails with a clear message if no checkpoint is available.

test:
	@echo "[TEST] Script: $(TRAIN_SCRIPT)"
	@ckpt_path="$(CKPT)"; \
	if [ -z "$$ckpt_path" ]; then \
	  ckpt_path="$(LATEST_CKPT)"; \
	  if [ -z "$$ckpt_path" ]; then \
	    echo "[ERROR] No checkpoint found under lightning_logs/$(EXPERIMENT_TAG). Provide CKPT=..."; \
	    exit 1; \
	  fi; \
	  echo "[TEST] Using latest checkpoint: $$ckpt_path"; \
	else \
	  echo "[TEST] Using provided checkpoint: $$ckpt_path"; \
	fi; \
	$(UV) run $(TRAIN_SCRIPT) --test-only --ckpt-path "$$ckpt_path" $(EXTRA)

# Convenience alias (some people type `make eval` instinctively)
.PHONY: eval

eval: test

save-experiments
	zip -r experiment_folders.zip figures/experiment_* -x "*.pyc" -x "__pycache__/*"