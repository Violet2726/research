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

Import-DotEnvLocal

$env:REPRO_MAX_CONCURRENT_REQUESTS = "$ReproMaxConcurrentRequests"
$env:REPRO_REQUESTS_PER_MINUTE_LIMIT = "$ReproRequestsPerMinuteLimit"
$env:REPRO_TOKENS_PER_MINUTE_LIMIT = "$ReproTokensPerMinuteLimit"

Write-Host "开始运行 reproduction_matrix full 阶段..."
Write-Host "使用限流: max_concurrent_requests=$ReproMaxConcurrentRequests, requests_per_minute_limit=$ReproRequestsPerMinuteLimit, tokens_per_minute_limit=$ReproTokensPerMinuteLimit"

$pythonScript = @'
import os

from research_experiments.matrix.faithful_matrix import RuntimeOverrides, assert_matrix_succeeded, run_matrix

run_dir = run_matrix(
    "reproduction",
    RuntimeOverrides(
        phase_name="full",
        max_concurrent_requests=int(os.environ["REPRO_MAX_CONCURRENT_REQUESTS"]),
        requests_per_minute_limit=int(os.environ["REPRO_REQUESTS_PER_MINUTE_LIMIT"]),
        tokens_per_minute_limit=int(os.environ["REPRO_TOKENS_PER_MINUTE_LIMIT"]),
    ),
)
assert_matrix_succeeded(run_dir)
print(run_dir.as_posix())
'@

$output = @($pythonScript | uv run python -)
if (-not $output) {
    throw "未能获取 full 阶段输出目录。"
}
$runDir = ($output | Select-Object -Last 1).Trim()
Write-Host "[$(Get-Date -Format s)] full 阶段完成: $runDir"
