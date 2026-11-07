$ErrorActionPreference = 'Stop'
$paperRoot = $PSScriptRoot
$paperBuild = Join-Path $paperRoot 'build'
New-Item -ItemType Directory -Path $paperBuild -Force | Out-Null
Push-Location -LiteralPath $paperRoot
try {
    foreach ($paperPass in 1..4) {
        $paperOutput = & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex 2>&1
        $paperExit = $LASTEXITCODE
        $paperOutput | Set-Content -LiteralPath (Join-Path $paperBuild "latex_pass$paperPass.txt") -Encoding utf8
        if ($paperExit -ne 0) {
            $paperOutput | Select-Object -Last 35
            throw "LaTeX compilation failed on pass $paperPass."
        }
        if ($paperPass -eq 1) {
            $paperBibOutput = & bibtex build/main 2>&1
            $paperBibExit = $LASTEXITCODE
            $paperBibOutput | Set-Content -LiteralPath (Join-Path $paperBuild 'bibtex_output.txt') -Encoding utf8
            if ($paperBibExit -ne 0) {
                $paperBibOutput | Write-Output
                throw 'BibTeX failed.'
            }
        }
    }
    Copy-Item -LiteralPath (Join-Path $paperBuild 'main.pdf') -Destination (Join-Path $paperRoot 'main.pdf')
    Write-Output "Built $(Join-Path $paperRoot 'main.pdf')"
    Select-String -LiteralPath (Join-Path $paperBuild 'main.aux') -Pattern 'newlabel\{main-text-end\}'
    Select-String -LiteralPath (Join-Path $paperBuild 'main.log') -Pattern 'Overfull|Underfull|LaTeX Warning|Package natbib Warning'
} finally {
    Pop-Location
}
