# Chimera — Extraction Progress

Two standalone repos have been extracted from chimera and published:
1. **[cs2-demo-viewer](https://github.com/skkwowee/cs2-demo-viewer)** — the Next.js demo viewer (from `site/`)
2. **[cs2-tools](https://github.com/skkwowee/cs2-tools)** — Python demo processing tools (from `src/` and `scripts/`)

---

## Viewer extraction (cs2-demo-viewer)

**Repo:** https://github.com/skkwowee/cs2-demo-viewer
**Tech:** Next.js 16, React 19, Tailwind v4

### Pipeline status
- [x] F01 — Extract site/ into standalone repo (git subtree split)
- [x] F02 — Rename and rebrand (package.json, metadata)
- [x] F03 — Add data format docs
- [x] F04 — Write README for standalone repo
- [x] F05 — Clean up chimera repo reference — README.md updated with links

Published at: https://github.com/skkwowee/cs2-demo-viewer

---

## Tools extraction (cs2-tools)

**Repo:** https://github.com/skkwowee/cs2-tools
**Dependency:** `cs2-tools[parse]` in `requirements.txt`

### Pipeline status
- [x] T01 — Inventory tools and fix imports
- [x] T02 — Create cs2-tools repo and copy files
- [x] T03 — Add pyproject.toml with optional deps
- [x] T04 — Write README for cs2-tools
- [x] T05 — Clean up chimera repo references — README.md updated with links

Published at: https://github.com/skkwowee/cs2-tools

---

## Decision logging convention

All significant technical decisions go in `docs/decisions.md` with sequential IDs (D001, D002, ...). Each entry includes: decision, rationale, alternatives considered, and status. Check current decision count with:

```bash
grep -c '^## D[0-9]' docs/decisions.md
```
