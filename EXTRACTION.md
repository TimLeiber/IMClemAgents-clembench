# Extraction provenance

Created 2026-09-09 from `/Users/tim/im_workspace/clembench` without modifying it.

- Source clembench HEAD: `94763255bede1d15404604a7a2fa6d326c7ac9b2`
- Local clemcore HEAD: `c11e2dd2dae6d14360b5783b8bdbd6f537ee7152`
- Both repositories may contain uncommitted changes; these hashes alone do not
  identify the full working environment
- Files were copied from the working tree, including game resources and instances
- macOS copy-on-write copies preserve independent files without hard-linking them
- Existing official and final recovery artifacts were copied without rewriting
- SAT_MENU results, legacy archives, root execution scripts and credentials were excluded
- No benchmark or inference was launched during extraction

Before publication: audit the external core dependency, curate historical helper
scripts and model configurations, choose artifact distribution, review licenses
and sensitive trace content. No remote repository was created.

## Registry curation

On 2026-09-09, model and agent entries were filtered against official result
directories containing interactions and the two retained recovery summaries.
The resulting registries contain seven models (including the Chronicle narrator)
and 24 agents. Excluded Gemma e4b server, 12b and 26b entries were removed only
from this extracted repository; the historical MLX e4b configuration remains.
Recovery reasoning overrides are recorded separately, without changing the
main experiment model configurations.

A small initial commit contained credential-free defaults and `.gitignore`.
A follow-up removes `key.json` from Git tracking while retaining the local file;
only the blank template is distributed. The README documents local setup.
The other extracted game, registry and analysis files remain uncommitted.
