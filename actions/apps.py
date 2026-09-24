"""
App launch actions. Non-blocking subprocess calls.
"""

import subprocess


def open_chrome():
    subprocess.Popen(["start", "chrome"], shell=True)


def open_spotify():
    subprocess.Popen(["start", "spotify"], shell=True)


def open_claude():
    """Launch the Claude desktop app (MSIX package) if installed, else fall back
    to Edge. `start claude` can't be used to detect failure: `start` always
    returns 0 even when the target doesn't resolve, so we check for the
    package explicitly instead of relying on a try/except around Popen."""
    app_id = _find_claude_app_id()
    if app_id:
        subprocess.Popen(
            ["powershell", "-Command", f"Start-Process 'shell:appsFolder\\{app_id}'"],
            shell=True,
        )
    else:
        subprocess.Popen(["start", "msedge", "https://claude.ai"], shell=True)


def _find_claude_app_id() -> str | None:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "(Get-StartApps | Where-Object {$_.Name -eq 'Claude'}).AppID"],
            capture_output=True, text=True, timeout=10,
        )
        app_id = result.stdout.strip()
        return app_id or None
    except Exception:
        return None


def open_edge():
    subprocess.Popen(["start", "msedge"], shell=True)


def open_vscode():
    subprocess.Popen(["start", "code"], shell=True)


def open_steam():
    subprocess.Popen(["start", "steam"], shell=True)


def open_all():
    """Claude app + Edge + VS Code simultaneously."""
    open_claude()
    open_edge()
    open_vscode()
