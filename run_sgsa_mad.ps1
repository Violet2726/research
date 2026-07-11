[CmdletBinding()]
param(
    [string]$Model = "dashscope/qwen-flash",
    [string[]]$Phases = @("count20_seed42", "count100_seed42")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Import-DotEnvLocal {
    param([string]$Path = ".env.local")
    if (-not (Test-Path -LiteralPath $Path)) { return }
    Get-Content -LiteralPath $Path -Encoding utf8 | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) { return }
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        Set-Item -Path "Env:$name" -Value $value
    }
}

Import-DotEnvLocal

$experimentPath = "configs/families/selective_gsa_mad/experiments/sgsa_mad.toml"

foreach ($phase in $Phases) {
    Write-Host "[$(Get-Date -Format s)] Running $experimentPath with model=$Model phase=$phase ..."
    $output = @(uv run research_cli experiment --family selective_gsa_mad run --experiment $experimentPath --phase $phase --model $Model)
    if (-not $output) {
        throw "Failed to run experiment with model=$Model phase=$phase."
    }
    $runDir = ($output | Select-Object -Last 1).Trim()
    Write-Host "[$(Get-Date -Format s)] Completed: $runDir"
}

Write-Host "[$(Get-Date -Format s)] All phases completed."
