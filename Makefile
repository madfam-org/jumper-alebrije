# Every step of the entry, from tests to the final video. Run from this directory.
# Assumes KingKong's `jumper` and `jumper-design` checkouts sit next to this one (see README).

PY ?= ../.venv/bin/python
SHELLFLOW = $(PY) ../jumper-design/scripts/shellflow.py
PROFILE = ../jumper-design/robots/jumper/profile.json
SKIN = skin/build/alebrije-jumper.skin
MAP = scene/build/dia-de-muertos-plaza.map
VIDEO = media/jumper-alebrije-dia-de-muertos.mp4

.PHONY: all test skin map verify dist video clean

all: test dist verify

# Every mode in KingKong's bundle is entered and nobody falls (about 2 min).
test:
	$(PY) sim/smoke_test.py

# The .skin, built and verified with jumper-design's own exporter.
skin:
	$(PY) skin/build_skin.py

# The .map, bundling the alebrije skin as its default robot.
map: skin
	$(PY) scene/build_map.py

# Re-run KingKong's validator on the release copies.
verify:
	$(SHELLFLOW) verify-package dist/alebrije-jumper.skin --profile $(PROFILE) --mujoco
	$(SHELLFLOW) verify-package dist/dia-de-muertos-plaza.map --profile $(PROFILE) \
		--capability rigid --capability jumper --mujoco

dist: map
	mkdir -p dist
	cp $(SKIN) $(MAP) dist/

# Simulate the choreography once, then render the 1080x1920 video (about 6 min).
video:
	mkdir -p out
	$(PY) sim/record.py choreo/take1.json out/take1.npz
	$(PY) render/video.py out/take1.npz choreo/take1.shots.json $(VIDEO)

clean:
	rm -rf out skin/build scene/build scene/textures MUJOCO_LOG.TXT
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
