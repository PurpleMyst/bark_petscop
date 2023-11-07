set windows-shell := ["pwsh", "-NoLogo", "-NoProfileLoadTime", "-Command"]
set dotenv-load

run-bark:
    cls && venv/Scripts/python.exe bark_petscop.py

run-frog:
    cls && venv/Scripts/python.exe frog_tts_petscop.py

format:
    black . -l100 -t py311 --preview
    isort .

install arg:
    venv/Scripts/python.exe -m pip install "{{arg}}"
