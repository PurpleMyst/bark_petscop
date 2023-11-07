set windows-shell := ["pwsh", "-NoLogo", "-NoProfileLoadTime", "-Command"]
set dotenv-load

run-bark:
    cls && venv/Scripts/python.exe bark_petscop.py

run-frog:
    cls && venv/Scripts/python.exe frog_tts_petscop.py

format:
    black .\bark_petscop.py -l100 -t py311 --preview
    isort .\bark_petscop.py

install arg:
    venv/Scripts/python.exe -m pip install "{{arg}}"
