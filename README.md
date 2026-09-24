# winnow
PII and credential detection for **Go codebases**


## Install

```bash
# 1. install uv, if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. from the repo root
cd winnow

# 3. create the venv (uv downloads python 3.12 itself if needed)
uv venv --python 3.12 .venv

# 4. install the project and its test deps
uv pip install -e ".[dev]"

# 5. activate 
source .venv/bin/activate
```

###shortcut
```
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install typer rich pathspec
PYTHONPATH=src python -m piiclf.cli scan ../go

or 
uv pip install -e ".[dev]"
piiclf scan <repo path>
piiclf scan <repo path> --json
piiclf scan <repo path> --raw
piiclf scan <repo path> --min-confidence 0.8


```
<img width="720" height="379" alt="Screenshot 2026-09-24 at 4 25 49 PM" src="https://github.com/user-attachments/assets/97b05759-cc05-49d0-ac7b-06cbf86e5ba6" />

