# Kaggriculture basedpyright LSP

This repo uses **basedpyright** (Cursor’s `cursorpyright` / Anysphere Python extension) pointed at the conda env `kagg`, which provides:

- `gymnasium`
- `stable_baselines3`
- `kaggle_environments`

## Files

| Path | Role |
|------|------|
| `pyproject.toml` | `[tool.basedpyright]` + `[tool.pyright]` (keep identical) |
| `pyrightconfig.json` | Same settings as `pyproject.toml` — **takes precedence** when both exist |
| `.vscode/settings.json` | Interpreter → conda `kagg` + `cursorpyright.analysis.extraPaths` |
| `scripts/kagg-basedpyright-lsp` | Custom langserver wrapper (`basedpyright-langserver --stdio`) |

**Layout:** main execution root `kaggriculture-self-training`; library lookup `datasets/scottweeden/self-training-code`; third-party imports from conda env `kagg`. Frozen `experiments/` snapshots are excluded from analysis.

## Verify

```bash
conda run -n kagg basedpyright -p . \
  datasets/scottweeden/self-training-code/kaggriculture_rl/_lsp_import_smoke.py
# expect: 0 errors
```

## Custom LSP client

```json
{
  "command": "/Users/sweeden/kagg/scripts/kagg-basedpyright-lsp",
  "args": ["--stdio"]
}
```

Reload the Cursor window after changing `pyrightconfig.json` so notebook import diagnostics refresh.
