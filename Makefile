NVCC ?= nvcc
NUM_B_LISTS ?= 4
BUILD_DIR ?= build/sm_89
VERSION ?= v7
RUNS ?= 5

NVCC_FLAGS ?= -O3 -std=c++17 -lineinfo
EXTRA_NVCC_FLAGS ?=

SOURCES := \
	src/v0.cu \
	src/v0_shared.cu \
	src/v1.cu \
	src/v1.5.cu \
	src/v2.cu \
	src/v3.cu \
	src/v4.cu \
	src/v5.cu \
	src/v6.cu \
	src/v7.cu

VERSIONS := $(basename $(notdir $(SOURCES)))
TARGETS := $(addprefix $(BUILD_DIR)/,$(VERSIONS))

.PHONY: all clean benchmark run list $(VERSIONS)

all: $(TARGETS)

$(VERSIONS): %: $(BUILD_DIR)/%

$(BUILD_DIR)/%: src/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -arch=sm_89 \
		-DNUM_B_LISTS=$(NUM_B_LISTS) $(EXTRA_NVCC_FLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $@

run: $(BUILD_DIR)/$(VERSION)
	$(BUILD_DIR)/$(VERSION)

benchmark: all
	BUILD_DIR=$(BUILD_DIR) ./scripts/benchmark.sh $(RUNS)

list:
	@printf '%s\n' $(VERSIONS)

clean:
	rm -rf build
