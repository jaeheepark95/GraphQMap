# CLAUDE.md

## Repo Overview

This repository contains multiple quantum initial-mapping branches. The two active sibling projects are:

- `src/kmw1` = canonical-hardware streamlined branch
- `src/kmw2` = v1.1-derived 5-channel branch

Do **not** mix their architecture, loss semantics, tensor conventions, or experiment assumptions unless the user explicitly asks for a comparison or port.

## Which CLAUDE file to use

Use a separate project-local `CLAUDE.md` for each active branch:

- `src/kmw1/CLAUDE.md` for `src/kmw1`(the streamlined canonical-hardware branch)
- `src/kmw2/CLAUDE.md` for `src/kmw2`(the v1.1-derived 5-channel branch)

Branch-specific files are the authority for implementation details.
This root file should stay short and only contain repo-wide rules. Project-specific details belong in the branch-local files.

## Startup Rule

When possible, launch Claude from the target project directory rather than the repo root.

Examples:

```bash
cd ~/KMWs_workspace/GraphQMap/KMW/src/kmw1
claude
```

```bash
cd ~/KMWs_workspace/GraphQMap/KMW/src/kmw2
claude
```

If Claude is launched from the repo root, it should still avoid mixing `kmw1` and `kmw2` context unless the user explicitly asks for cross-branch analysis.

## Shared Environment

Use:

```bash
conda activate graphqmap_pascal
cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"
```

## Shared Critical Rules

1. Always identify which branch the task targets before making code or design claims.
2. Do not assume `kmw1` and `kmw2` share the same representation, loss family, or training schedule.
3. Preserve existing manifest split semantics unless the user explicitly asks to regenerate or redesign splits.
4. Prefer reading the project-local `CLAUDE.md` plus the minimum relevant code files rather than scanning the whole repo.
5. For architecture-changing work, read the target branch’s main design spec before proposing changes.

## Branch Routing Summary

### `kmw1`
- streamlined canonical-hardware branch
- no learned reindexer
- hardware-only canonicalization
- single-matrix circuit input + hardware-token conditioning

### `kmw2`
- v1.1-derived 5-channel branch
- separate preprocessing helpers (`extractor`, `canonical_indexer`, `featurizer`, `pipeline`)
- staged training over existing source manifests
- explicit in-package PST metric

## Minimal Repo-Level Read Set

Only read these from the repo root by default:

1. this root `CLAUDE.md`
2. the target branch’s local `CLAUDE.md`
3. the target branch’s main design spec

Do not automatically scan sibling branches.
