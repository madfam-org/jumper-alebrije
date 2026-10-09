# Every step, for any character: `make <target> CHAR=<id>` (default: alebrije).
# Run from this directory, with KingKong's `jumper` and `jumper-design` beside it (see README).

PY ?= ../.venv/bin/python
JK = $(PY) -m jumperkit
CHAR ?= alebrije
TAKE = build/takes/take1.npz

.PHONY: help test regress check look stills skin map package dist verify video lineup clean

help:
	@grep -E '^[a-z]+:.*## ' Makefile | sed -E 's/:.*## /\t/'

test: ## fast tests: specs valid, looks display-only, alebrije unchanged (seconds)
	$(PY) -m pytest

regress: ## slow: rebuild the released alebrije packages, run every mode (minutes)
	$(PY) -m pytest -m slow

check: ## validate every character spec and list its decorations
	$(JK) check

look: ## 4-view studio sheet of CHAR -> build/CHAR/look.png (seconds)
	$(JK) look -c $(CHAR)

stills: ## CHAR in the plaza from 3 angles -> build/CHAR/plaza.png
	$(JK) stills -c $(CHAR)

skin: ## build + verify CHAR's .skin
	$(JK) skin -c $(CHAR)

map: skin ## build + verify CHAR's .map (bundles the skin)
	$(JK) map -c $(CHAR)

package: ## everything for a release of CHAR: packages, thumbnails, dist/, verify
	$(JK) skin -c $(CHAR)
	$(JK) map -c $(CHAR)
	$(JK) skin-preview -c $(CHAR)
	$(JK) map-preview -c $(CHAR)
	$(JK) skin -c $(CHAR)
	$(JK) map -c $(CHAR)
	$(JK) dist -c $(CHAR)
	$(JK) verify -c $(CHAR)

dist: map ## copy CHAR's .skin and .map into dist/
	$(JK) dist -c $(CHAR)

verify: ## re-run KingKong's verify-package on CHAR's dist/ files
	$(JK) verify -c $(CHAR)

$(TAKE): choreo/take1.json
	$(JK) record choreo/take1.json $@

video: $(TAKE) ## the 43 s vertical video of CHAR -> build/CHAR/video.mp4 (~6 min)
	$(JK) video -c $(CHAR)

lineup: $(TAKE) ## every character at the same frame -> build/lineup/lineup.png
	$(JK) lineup

clean: ## remove generated files (build/); dist/ and media/ are kept
	rm -rf build MUJOCO_LOG.TXT
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
