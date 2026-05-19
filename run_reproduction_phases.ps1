$ErrorActionPreference = "Stop"

$ReproMaxConcurrentRequests = 80
$ReproRequestsPerMinuteLimit = 95
$ReproTokensPerMinuteLimit = 1000000

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

function Invoke-ReproductionPhase {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Phase,
        [string]$ReferenceStatePath = ""
    )

    $env:REPRO_PHASE = $Phase
    $env:REPRO_MAX_CONCURRENT_REQUESTS = "$ReproMaxConcurrentRequests"
    $env:REPRO_REQUESTS_PER_MINUTE_LIMIT = "$ReproRequestsPerMinuteLimit"
    $env:REPRO_TOKENS_PER_MINUTE_LIMIT = "$ReproTokensPerMinuteLimit"
    if ($ReferenceStatePath) {
        $env:REPRO_REFERENCE_STATE = $ReferenceStatePath
    } else {
        Remove-Item -Path Env:REPRO_REFERENCE_STATE -ErrorAction SilentlyContinue
    }

    $pythonScript = @'
import os

from research_experiments.matrix.faithful_matrix import RuntimeOverrides, assert_matrix_succeeded, run_matrix

kwargs = {}
reference_state = os.environ.get("REPRO_REFERENCE_STATE")
if reference_state:
    kwargs["reference_state_path_or_root"] = reference_state

run_dir = run_matrix(
    "reproduction",
    RuntimeOverrides(
        phase_name=os.environ["REPRO_PHASE"],
        max_concurrent_requests=int(os.environ["REPRO_MAX_CONCURRENT_REQUESTS"]),
        requests_per_minute_limit=int(os.environ["REPRO_REQUESTS_PER_MINUTE_LIMIT"]),
        tokens_per_minute_limit=int(os.environ["REPRO_TOKENS_PER_MINUTE_LIMIT"]),
    ),
    **kwargs,
)
assert_matrix_succeeded(run_dir)
print(run_dir.as_posix())
'@

    $output = @($pythonScript | uv run python -)
    if (-not $output) {
        throw "未能获取阶段输出目录。"
    }
    return ($output | Select-Object -Last 1).Trim()
}

Import-DotEnvLocal

Write-Host "开始运行 reproduction_matrix 三个阶段..."
Write-Host "使用限流: max_concurrent_requests=$ReproMaxConcurrentRequests, requests_per_minute_limit=$ReproRequestsPerMinuteLimit, tokens_per_minute_limit=$ReproTokensPerMinuteLimit"

Write-Host "[$(Get-Date -Format s)] 开始运行 count20 阶段..."
$count20Dir = Invoke-ReproductionPhase -Phase "count20"
Write-Host "[$(Get-Date -Format s)] count20 阶段完成: $count20Dir"

Write-Host "[$(Get-Date -Format s)] 开始运行 count100 阶段..."
$count100Dir = Invoke-ReproductionPhase -Phase "count100" -ReferenceStatePath $count20Dir
Write-Host "[$(Get-Date -Format s)] count100 阶段完成: $count100Dir"

Write-Host "[$(Get-Date -Format s)] 开始运行 count300 阶段..."
$count300Dir = Invoke-ReproductionPhase -Phase "count300" -ReferenceStatePath $count100Dir
Write-Host "[$(Get-Date -Format s)] count300 阶段完成: $count300Dir"

$autoPushCache = if ([string]::IsNullOrWhiteSpace($env:RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT)) { "" } else { $env:RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT.ToLowerInvariant() }
if (
    ($autoPushCache -in @("1", "true", "yes", "on")) -and
    -not [string]::IsNullOrWhiteSpace($env:RESEARCH_CACHE_HF_REPO)
) {
    $cacheRoot = if ([string]::IsNullOrWhiteSpace($env:RESEARCH_CACHE_ROOT)) { "local/cache" } else { $env:RESEARCH_CACHE_ROOT }
    Write-Host "[$(Get-Date -Format s)] 开始推送 cache 最新快照到 Hugging Face: $cacheRoot"
    $pushOutput = uv run cache_archive_cli push-latest --cache-root $cacheRoot --repo $env:RESEARCH_CACHE_HF_REPO --json
    $pushSummary = ($pushOutput -join "`n") | ConvertFrom-Json
    Write-Host "[$(Get-Date -Format s)] cache 快照推送完成: $($pushSummary.remote_repo)"
}

Write-Host "[$(Get-Date -Format s)] 所有阶段运行完成。"
