param(
    [string]$InputDirectory,
    [string]$OutputPath,
    [string]$Filter = "*.csv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $InputDirectory)) {
    throw "Input directory does not exist: $InputDirectory"
}

$files = Get-ChildItem -Path $InputDirectory -Filter $Filter | Sort-Object Name
if (-not $files) {
    throw "No files matched '$Filter' in $InputDirectory"
}

$outputDirectory = Split-Path -Parent $OutputPath
if ($outputDirectory) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}
if (Test-Path $OutputPath) {
    Remove-Item $OutputPath -Force
}

$wroteHeader = $false
$lineCount = 0

foreach ($file in $files) {
    $isFirstLine = $true
    Get-Content -Path $file.FullName | ForEach-Object {
        if ($isFirstLine) {
            $isFirstLine = $false
            if ($wroteHeader) {
                return
            }
            $wroteHeader = $true
        }
        Add-Content -Path $OutputPath -Value $_
        $lineCount++
    }
}

Write-Host ("Merged {0} files into {1}" -f $files.Count, $OutputPath)
Write-Host ("Lines written: {0}" -f $lineCount)
