# Sanity check script for Linkjiru cpp_impl
# Uses CLion's bundled clang-format and clang-tidy (no LLVM install needed)

$clangFormat = "C:\Program Files\JetBrains\CLion 2025.3.2\plugins\clion-radler\DotFiles\windows-x64\clang-format.exe"
$clangTidy = "C:\Program Files\JetBrains\CLion 2025.3.2\bin\clang\win\x64\bin\clang-tidy.exe"

$srcDir = "$PSScriptRoot\..\src\linkjiru"
$buildDir = "$PSScriptRoot\..\cmake-build-release"

Write-Host "`n=== clang-format (dry run) ===" -ForegroundColor Cyan
$files = Get-ChildItem -Path $srcDir -Include *.h, *.cpp -Recurse
$formatIssues = 0
foreach ($file in $files) {
    $result = & $clangFormat --dry-run --Werror $file.FullName 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [FORMAT] $($file.Name)" -ForegroundColor Yellow
        $formatIssues++
    }
}
if ($formatIssues -eq 0) {
    Write-Host "  All files formatted correctly" -ForegroundColor Green
} else {
    Write-Host "  $formatIssues file(s) need formatting. Run with -Fix to apply." -ForegroundColor Yellow
}

if ($args -contains "-Fix") {
    Write-Host "`n=== clang-format (applying fixes) ===" -ForegroundColor Cyan
    foreach ($file in $files) {
        & $clangFormat -i $file.FullName
    }
    Write-Host "  Done" -ForegroundColor Green
}

Write-Host "`n=== clang-tidy ===" -ForegroundColor Cyan
$compileDb = "$buildDir\compile_commands.json"
if (-not (Test-Path $compileDb)) {
    Write-Host "  compile_commands.json not found at $compileDb" -ForegroundColor Red
    Write-Host "  Run cmake configure first (CMAKE_EXPORT_COMPILE_COMMANDS is ON)" -ForegroundColor Red
    exit 1
}

$tidyIssues = 0
$cppFiles = Get-ChildItem -Path $srcDir -Filter *.cpp -Recurse
foreach ($file in $cppFiles) {
    Write-Host "  Checking $($file.Name)..." -ForegroundColor Gray
    $output = & $clangTidy $file.FullName -p $buildDir 2>&1
    foreach ($line in $output) {
        # Only show warnings/errors from our source files, not from Boost/JUCE/deps
        if ($line -match "src[\\/]linkjiru[\\/]" -and $line -match "warning:|error:") {
            Write-Host "    $line" -ForegroundColor Yellow
            $tidyIssues++
        }
    }
}

if ($tidyIssues -eq 0) {
    Write-Host "  No issues found in src/linkjiru/" -ForegroundColor Green
} else {
    Write-Host "  $tidyIssues issue(s) found" -ForegroundColor Yellow
}

Write-Host "`n=== Done ===" -ForegroundColor Cyan
