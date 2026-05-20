# igg2 Task List

## Status: COMPLETE — T11 still blocked on DinD seccomp; all other tasks including T17/T18 improvements done (39 tests pass)

- [x] T16: Added igg2/data/.gitkeep + igg2/LICENSE (MIT, 2026)

## Phase 1 — Core Algorithm + CLI

- [x] T01: Create project skeleton (dirs, __init__.py, sample.csv fixture)
- [x] T02: Implement src/tokenize.py (regex tokeniser, no NLTK)
- [x] T03: Implement src/cluster.py (TF-IDF + MiniBatchKMeans → topic per row)
- [x] T04: Implement src/train.py (CSV → topic-aware JSON model)
- [x] T05: Implement src/generate.py (JSON model → synthesised rows)
- [x] T06: Implement src/cli.py (train / generate / inspect commands)
- [x] T07: Write tests (test_cluster, test_train, test_generate, test_coherence) — 27 passed

## Phase 2 — Docker + Build

- [x] T08: Write Pipfile
- [x] T09: Write Dockerfile, docker-compose.yml, Makefile
- [x] T10: Generate Pipfile.lock (host-side pipenv lock)
- [ ] T11: BLOCKED — Docker DinD blocked by seccomp (CLONE_NEWNS denied). Files are written and manually validated; needs a privileged/DinD-capable environment to actually `docker build` and run `make test` inside the container.

## Phase 3 — Tuning Knobs

- [x] T12: Sidecar YAML support in train.py + cli.py (load_sidecar + cli.py already done)
- [x] T13: Topic labels in igg2 inspect (already done in train.py)

## Cleanup / Review

- [x] T14: Security review — fixed SSRF (URL rejection in train+sidecar), non-root Docker user
- [x] T15: README written at igg2/README.md

## Phase 4 — Incremental Improvements

- [x] T17: Add test_tokenize.py — only source module with zero direct unit tests; covers tokenize(), STOP_WORDS, edge cases (empty, special chars, numbers, stop-word filtering)
- [x] T18: Fix source_file path disclosure — train.py stores os.path.abspath; change to basename to match spec and avoid leaking filesystem paths in shareable model JSON

## Work Files

- `.ralph/prompt.md` — original spec
- `.ralph/tasks.md` — this file
- `igg2/` — project root
- `.tmp/model/*.csv` — sample CSV fixtures
