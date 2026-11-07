$ErrorActionPreference = 'Stop'
$taskFigureRoot = $PSScriptRoot
$taskPaperRoot = Split-Path -Parent $taskFigureRoot
$taskBuildDir = Join-Path $taskFigureRoot 'build'
New-Item -ItemType Directory -Force -Path $taskBuildDir | Out-Null
Push-Location -LiteralPath $taskPaperRoot
try {
    $taskFigureNames = @(
        'fig01_concept', 'fig02_learning', 'fig03_assessment', 'fig04_revelation',
        'supp_frontier', 'supp_diagnostics', 'supp_sensitivity', 'supp_coverage',
        'supp_revelation'
    )
    foreach ($taskFigureName in $taskFigureNames) {
        $taskOutput = & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error `
            -output-directory=figures/build "figures/$taskFigureName.tex" 2>&1
        if ($LASTEXITCODE -ne 0) {
            $taskOutput | Select-Object -Last 35
            throw "Figure compilation failed: $taskFigureName"
        }
        Copy-Item -LiteralPath (Join-Path $taskBuildDir "$taskFigureName.pdf") `
            -Destination (Join-Path $taskFigureRoot "$taskFigureName.pdf")
        Write-Output "Built $taskFigureName.pdf"
    }
} finally {
    Pop-Location
}
