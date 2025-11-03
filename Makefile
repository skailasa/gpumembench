# ============================================================
# GPU backend selector
# Usage:
#   make BACKEND=cuda
#   make BACKEND=amd
# Default: none (no GPU build)
# ============================================================

BACKEND ?= none

# Directories per backend
CUDA_DIRS = cachebench-cuda shmembench-cuda
AMD_DIRS  = cachebench-amd  shmembench-amd

# Binaries per backend
CUDA_BINARIES = \
	cachebench-cuda/cachebench \
	cachebench-cuda/cachebench-l2-only \
	cachebench-cuda/cachebench-tex-loads \
	shmembench-cuda/shmembench \

AMD_BINARIES = shmembench-amd/shmembench

# ============================================================
# Conditional logic for backends
# ============================================================

ifeq ($(BACKEND),cuda)

all:
	@echo ">>> Building CUDA benchmarks..."
	$(foreach dir,$(CUDA_DIRS),$(MAKE) -C $(dir);)
	mkdir -p bin/cuda
	cp $(CUDA_BINARIES) bin/cuda/

clean:
	$(foreach dir,$(CUDA_DIRS),$(MAKE) -C $(dir) clean;)

rebuild:
	$(foreach dir,$(CUDA_DIRS),$(MAKE) -C $(dir) rebuild;)

else ifeq ($(BACKEND),amd)

all:
	@echo ">>> Building AMD benchmarks..."
	$(foreach dir,$(AMD_DIRS),$(MAKE) -C $(dir);)
	mkdir -p bin/amd
	cp $(AMD_BINARIES) bin/amd/

clean:
	$(foreach dir,$(AMD_DIRS),$(MAKE) -C $(dir) clean;)

rebuild:
	$(foreach dir,$(AMD_DIRS),$(MAKE) -C $(dir) rebuild;)

else

all:
	@echo ">>> No backend specified."
	@echo "    Use one of:"
	@echo "      make BACKEND=cuda"
	@echo "      make BACKEND=amd"

clean:
	@echo ">>> No backend specified. Nothing to clean."

rebuild:
	@echo ">>> No backend specified. Nothing to rebuild."

endif
