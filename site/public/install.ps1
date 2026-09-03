# Friday install bootstrap for Windows.   https://friday.palash.dev/install.ps1
#
#   irm https://friday.palash.dev/install.ps1 | iex
#
# Friday runs on Linux x86_64 with an NVIDIA GPU. It does not run natively on
# Windows. This script finds an existing WSL 2 distribution, confirms with
# you, and runs the Linux bootstrap (https://friday.palash.dev/install) inside
# it. That bootstrap downloads the newest versioned installer and verifies it
# against the release's SHA256SUMS before running it.
#
# WSL 2 is not a qualified Friday target. The Linux installer still runs its
# platform, systemd user-session, GPU, disk, and asset checks and stops if
# they fail. You will need:
#   - WSL 2 with an x86_64 distribution and systemd enabled
#     ([boot] systemd=true in /etc/wsl.conf inside the distribution)
#   - The Windows NVIDIA driver with CUDA support for WSL 2
#   - About 50 GiB free inside the distribution
# Desktop actions need Hyprland and are unavailable inside WSL.
#
# Environment:
#   FRIDAY_WSL_DISTRO   run inside this distribution instead of the default
#   FRIDAY_VERSION      install this tag (for example v0.1.0-alpha.1)
#   FRIDAY_ASSUME_YES   set to 1 to skip the confirmation prompt

$ErrorActionPreference = 'Stop'

function Write-Note([string] $Message) {
    Write-Host "friday bootstrap: $Message"
}

function Stop-Bootstrap([string] $Message) {
    Write-Host "friday bootstrap: $Message" -ForegroundColor Red
    exit 1
}

$onWindows = ($PSVersionTable.PSVersion.Major -lt 6) -or $IsWindows
if (-not $onWindows) {
    Stop-Bootstrap 'this script is for Windows. On Linux run: curl -fsSL https://friday.palash.dev/install | bash'
}

$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wsl) {
    Write-Note 'Friday needs a Linux environment. WSL 2 is not installed.'
    Write-Note 'Install it from an administrator PowerShell with:  wsl --install'
    Write-Note 'Reboot, finish the distribution setup, then run this command again.'
    exit 1
}

# wsl.exe writes UTF-16 output; read it as such so names are not mangled.
$previousEncoding = [Console]::OutputEncoding
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
    $distributions = @(& wsl.exe --list --quiet 2>$null |
        ForEach-Object { ($_ -replace "`0", '').Trim() } |
        Where-Object { $_ -ne '' })
} finally {
    [Console]::OutputEncoding = $previousEncoding
}

if ($distributions.Count -eq 0) {
    Write-Note 'WSL is installed but no distribution is set up.'
    Write-Note 'Install one from an administrator PowerShell, for example:  wsl --install -d Ubuntu'
    exit 1
}

$distroArgs = @()
$distro = $env:FRIDAY_WSL_DISTRO
if ($distro) {
    if ($distributions -notcontains $distro) {
        Stop-Bootstrap "WSL distribution '$distro' was not found. Available: $($distributions -join ', ')"
    }
    $distroArgs = @('-d', $distro)
} else {
    $distro = 'the default WSL distribution'
}

$command = 'curl -fsSL https://friday.palash.dev/install | bash'
if ($env:FRIDAY_VERSION) {
    if ($env:FRIDAY_VERSION -notmatch '^v[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$') {
        Stop-Bootstrap 'FRIDAY_VERSION must be a release tag such as v0.1.0-alpha.1'
    }
    $command = "FRIDAY_VERSION=$($env:FRIDAY_VERSION) $command"
}

$machine = (& wsl.exe @distroArgs -- uname -m 2>$null | ForEach-Object { $_.Trim() }) -join ''
if ($machine -ne 'x86_64') {
    Stop-Bootstrap "Friday requires an x86_64 distribution; $distro reports '$machine'"
}
& wsl.exe @distroArgs -- sh -c 'command -v curl >/dev/null 2>&1' 2>$null
if ($LASTEXITCODE -ne 0) {
    Stop-Bootstrap "curl is not installed inside $distro. Install it there first (for example: sudo apt install curl)."
}

Write-Note "Friday will be installed inside $distro."
Write-Note 'WSL 2 is not a qualified target; the Linux installer will check systemd, GPU, and disk.'
if ($env:FRIDAY_ASSUME_YES -ne '1') {
    $answer = Read-Host 'Continue? [y/N]'
    if ($answer -notmatch '^[Yy]') {
        Write-Note 'cancelled; nothing was changed'
        exit 0
    }
}

& wsl.exe @distroArgs -- bash -c $command
exit $LASTEXITCODE
