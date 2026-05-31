ARCH ?= 90;120

.PHONY: build bench profile analyze capture clean
build:           ; cmake -B build -DCMAKE_CUDA_ARCHITECTURES="$(ARCH)" && cmake --build build -j
bench: build     ; bash scripts/run_bench.sh
profile: build   ; bash scripts/profile.sh
analyze:         ; python analysis/plot.py
capture: bench profile analyze   ## full unattended capture -> results/
clean:           ; rm -rf build results/bench.csv results/*.png results/*.rep results/report.md
