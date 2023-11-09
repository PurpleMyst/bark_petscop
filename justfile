set windows-shell := ["pwsh", "-NoLogo", "-NoProfileLoadTime", "-Command"]
set dotenv-load

run +args:
    venv/Scripts/python.exe frog_tts_petscop.py {{args}}

format:
    ruff format .
    isort .

install arg:
    venv/Scripts/python.exe -m pip install "{{arg}}"
