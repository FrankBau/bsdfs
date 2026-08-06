# Experiments for "Enumerating Length-Bounded Simple Paths and Cycles ..."
#
#   make              run every experiment that is out of date
#   make runtime      run a single experiment
#   make -B           force a full rerun
#
# Each experiment writes <name>.pdf (figure) and <name>.txt (full log,
# including the per-instance lines and the totals tables).

PYTHON  ?= python3
PYFLAGS ?= -u -OO

EXPERIMENTS = missed_paths steps runtime delay_bounds
PDFS = $(addsuffix .pdf,$(EXPERIMENTS))

export PYTHONIOENCODING = utf-8

.PHONY: experiments clean distclean $(EXPERIMENTS)
.NOTPARALLEL:          # runtime.py measures wall clock: never run in parallel
.DELETE_ON_ERROR:      # a failed run must not leave a stale figure behind

experiments: $(PDFS)

$(EXPERIMENTS): %: %.pdf

%.pdf: %.py
	@echo "=== $* started  $$(date -Is)"
	@start=$$(date +%s); \
	{ $(PYTHON) -c "import sys,platform;print('#',platform.platform(),sys.version)" ; \
	  echo "# $* $$(date -Is)"; \
	  echo "# $$(lscpu | sed -n 's/^Model name: *//p') | $$(nproc) cores | $$(free -h | awk '/^Mem:/{print $$2}') RAM"; \
	  echo "# $$(. /etc/os-release; echo $$PRETTY_NAME) | $$($(PYTHON) -V)"; \
	  $(PYTHON) $(PYFLAGS) $<; \
	  echo "# $* done, $$(( $$(date +%s) - start )) s"; \
	} > $*.txt
	@echo "=== $* finished $$(date -Is) -- $$(tail -1 $*.txt)"

# only these two import the reference implementations
incompleteness.pdf runtime.pdf: bsdfs.py bcdfs.py

clean:
	rm -rf __pycache__

distclean: clean
	rm -f $(PDFS) $(addsuffix .txt,$(EXPERIMENTS))
