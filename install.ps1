<#
  install.ps1 - one-shot setup for Fairy + Hermes Agent on Windows.

  What this does, in order:
    1. Installs Ollama if it isn't already on PATH (via winget).
    2. Pulls the local models Fairy expects: gemma4, qwen2.5-coder:7b.
    3. Installs Hermes Agent using NousResearch's own official installer
       (this script does NOT vendor a copy of hermes-agent - it always
       fetches the current release straight from the upstream project).
    4. Creates a Python virtual environment for Fairy and installs
       requirements.txt into it.
    5. Copies the config templates to their real filenames if they don't
       exist yet, so you just have to fill in values, not create files.

  This makes real changes to your system (installs software, downloads
  multi-GB model weights). Read it before running it. Run from an ordinary
  PowerShell prompt in the repo root:

      .\install.ps1

  Some steps (winget package installs) may prompt for elevation.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# --- 1. Ollama -----------------------------------------------------------
Write-Step "Checking for Ollama"
if (Test-Command "ollama") {
    Write-Host "Ollama is already installed."
} else {
    if (Test-Command "winget") {
        Write-Host "Installing Ollama via winget..."
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    } else {
        Write-Host "winget not found. Install Ollama manually from https://ollama.com/download and re-run this script." -ForegroundColor Yellow
        exit 1
    }
}

# --- 2. Pull models --------------------------------------------------------
Write-Step "Pulling local models (this can take a while - multi-GB downloads)"
ollama pull gemma4
ollama pull qwen2.5-coder:7b

# --- 3. Hermes Agent ---------------------------------------------------
Write-Step "Installing Hermes Agent (official installer, from Nous Research)"
if (Test-Command "hermes") {
    Write-Host "Hermes Agent already appears to be installed (found 'hermes' on PATH)."
} else {
    try {
        Invoke-Expression (Invoke-RestMethod https://hermes-agent.nousresearch.com/install.ps1)
    } catch {
        Write-Host "Hermes Agent install failed or was skipped: $_" -ForegroundColor Yellow
        Write-Host "You can retry manually later with:"
        Write-Host '  iex (irm https://hermes-agent.nousresearch.com/install.ps1)'
    }
}

# --- 4. Fairy's own Python environment -----------------------------------
Write-Step "Setting up Fairy's Python virtual environment"
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".\.venv\Scripts\pip.exe" install --upgrade pip
& ".\.venv\Scripts\pip.exe" install -r requirements.txt

# --- 5. Config templates --------------------------------------------------
Write-Step "Setting up config files from templates"
if (-not (Test-Path "config\api_keys.json")) {
    Copy-Item "config\api_keys.example.json" "config\api_keys.json"
    Write-Host "Created config\api_keys.json - fill in your API keys."
} else {
    Write-Host "config\api_keys.json already exists, leaving it alone."
}
if (-not (Test-Path "controller\.env")) {
    Copy-Item "controller\.env.example" "controller\.env"
    Write-Host "Created controller\.env - fill in your tokens/keys."
} else {
    Write-Host "controller\.env already exists, leaving it alone."
}

Write-Step "Done"
Write-Host "Next steps:"
Write-Host "  1. Fill in config\api_keys.json and controller\.env"
Write-Host "  2. .\.venv\Scripts\activate"
Write-Host "  3. python main.py     (text REPL)   or   python fairy.py   (TUI + voice)"
