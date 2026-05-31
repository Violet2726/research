[CmdletBinding()]
param(
    # [string[]]$Phases = @("count20", "count100", "count300", "count500"),
    [string[]]$Phases = @("count20"),
    [string]$InitialReferenceStatePath = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Import-DotEnvLocal {
    param(
        [string]$Path = ".env.local"
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    Get-Content -LiteralPath $Path -Encoding utf8 | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }
        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) {
            return
        }
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        Set-Item -Path "Env:$name" -Value $value
    }
}

function Invoke-FaithfulPhase {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Phase,
        [string]$ReferenceStatePath = ""
    )

    $cliArgs = @("research_cli", "matrix", "run", "--matrix", "faithful", "--phase", $Phase)
    if (-not [string]::IsNullOrWhiteSpace($ReferenceStatePath)) {
        $cliArgs += @("--reference-state-path", $ReferenceStatePath)
    }

    $output = @(uv run @cliArgs)
    if (-not $output) {
        throw "Failed to resolve matrix run directory."
    }

    $runDir = ($output | Select-Object -Last 1).Trim()
    @(uv run research_cli matrix assert-success --state-path $runDir --json) | Out-Null
    return $runDir
}

function Invoke-OptionalCachePush {
    $autoPushCache = if ([string]::IsNullOrWhiteSpace($env:RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT)) {
        ""
    } else {
        $env:RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT.ToLowerInvariant()
    }
    if (
        ($autoPushCache -notin @("1", "true", "yes", "on")) -or
        [string]::IsNullOrWhiteSpace($env:RESEARCH_CACHE_HF_REPO)
    ) {
        return
    }

    $cacheRoot = if ([string]::IsNullOrWhiteSpace($env:RESEARCH_CACHE_ROOT)) {
        "local/cache"
    } else {
        $env:RESEARCH_CACHE_ROOT
    }
    Write-Host "[$(Get-Date -Format s)] Pushing latest cache snapshot to Hugging Face: $cacheRoot"
    $pushOutput = uv run research_cli tools cache-archive push-latest --cache-root $cacheRoot --repo $env:RESEARCH_CACHE_HF_REPO --json
    $pushSummary = ($pushOutput -join "`n") | ConvertFrom-Json
    Write-Host "[$(Get-Date -Format s)] Cache snapshot push completed: $($pushSummary.remote_repo)"
}

Import-DotEnvLocal

Write-Host "Starting faithful_matrix phase sequence..."

$previousRunDir = $InitialReferenceStatePath
foreach ($phase in $Phases) {
    Write-Host "[$(Get-Date -Format s)] Starting phase $phase ..."
    if ([string]::IsNullOrWhiteSpace($previousRunDir)) {
        $previousRunDir = Invoke-FaithfulPhase -Phase $phase
    } else {
        $previousRunDir = Invoke-FaithfulPhase -Phase $phase -ReferenceStatePath $previousRunDir
    }
    Write-Host "[$(Get-Date -Format s)] Phase $phase completed: $previousRunDir"
}

Invoke-OptionalCachePush

Write-Host "[$(Get-Date -Format s)] All phases completed."
