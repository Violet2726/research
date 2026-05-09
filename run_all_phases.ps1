$ErrorActionPreference = "Stop"

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

function Invoke-MatrixPhase {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Phase,
        [string]$ReferenceStatePath = ""
    )

    $env:FAITHFUL_PHASE = $Phase
    $env:FAITHFUL_STATE_ROOT = "runs/faithful_matrix_iterative"
    if ($ReferenceStatePath) {
        $env:FAITHFUL_REFERENCE_STATE = $ReferenceStatePath
    } else {
        Remove-Item -Path Env:FAITHFUL_REFERENCE_STATE -ErrorAction SilentlyContinue
    }

    $pythonScript = @'
import os

from experiment_core.matrix.faithful_matrix import RuntimeOverrides, run_faithful_matrix

kwargs = {"state_root": os.environ["FAITHFUL_STATE_ROOT"]}
reference_state = os.environ.get("FAITHFUL_REFERENCE_STATE")
if reference_state:
    kwargs["reference_state_path_or_root"] = reference_state

run_dir = run_faithful_matrix(
    RuntimeOverrides(phase_name=os.environ["FAITHFUL_PHASE"]),
    **kwargs,
)
print(run_dir.as_posix())
'@

    $output = @($pythonScript | uv run python -)
    if (-not $output) {
        throw "未能获取阶段输出目录。"
    }
    return ($output | Select-Object -Last 1).Trim()
}

Import-DotEnvLocal

Write-Host "开始运行 faithful_matrix 三个阶段..."

Write-Host "[$(Get-Date -Format s)] 开始运行 smoke20 阶段..."
$smoke20Dir = Invoke-MatrixPhase -Phase "smoke20"
Write-Host "[$(Get-Date -Format s)] smoke20 阶段完成: $smoke20Dir"

Write-Host "[$(Get-Date -Format s)] 开始运行 pilot100 阶段..."
$pilot100Dir = Invoke-MatrixPhase -Phase "pilot100" -ReferenceStatePath $smoke20Dir
Write-Host "[$(Get-Date -Format s)] pilot100 阶段完成: $pilot100Dir"

Write-Host "[$(Get-Date -Format s)] 开始运行 confirmatory300 阶段..."
$confirmatory300Dir = Invoke-MatrixPhase -Phase "confirmatory300" -ReferenceStatePath $pilot100Dir
Write-Host "[$(Get-Date -Format s)] confirmatory300 阶段完成: $confirmatory300Dir"

Write-Host "[$(Get-Date -Format s)] 所有阶段运行完成。"
