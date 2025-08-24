# Claude Code Build Integration Command for Unit Tests
# This script builds the unit test project and automatically requests Claude to fix errors

param(
    [string]$Action = "build"
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$MSBuild = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
if (-not (Test-Path $MSBuild)) {
    $MSBuild = "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe"
}
if (-not (Test-Path $MSBuild)) {
    $MSBuild = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
}

function Invoke-Build {
    Write-Host "Building unit tests..." -ForegroundColor Cyan
    
    # Clean previous error file
    if (Test-Path "build_errors.txt") { Remove-Item "build_errors.txt" }
    
    # Build
    & $MSBuild builds\VisualStudio2022\UnitTests.sln /p:Configuration=Debug /p:Platform=x64 /v:minimal /fl "/flp:logfile=build_errors.txt;errorsonly;verbosity=detailed"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nBUILD SUCCEEDED!" -ForegroundColor Green
        return $true
    }
    else {
        Write-Host "`nBUILD FAILED" -ForegroundColor Red
        
        # Format errors for Claude
        if (Test-Path "build_errors.txt") {
            $content = Get-Content "build_errors.txt" -Raw
            $formatted = "=== Unit Test Build Errors for Claude ===" + "`n"
            $formatted += "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" + "`n"
            $formatted += "=" * 50 + "`n`n"
            $formatted += $content
            $formatted | Out-File -FilePath "build_errors.txt" -Encoding UTF8
        }
        
        return $false
    }
}

# Main action
switch ($Action.ToLower()) {
    "build" {
        $success = Invoke-Build
        if (-not $success) {
            Write-Host "`nErrors detected. Ready for Claude to fix them." -ForegroundColor Yellow
            Write-Host "Use: read unittest/build_errors.txt and fix the errors" -ForegroundColor Cyan
        }
    }
    "loop" {
        # Build loop mode
        do {
            $success = Invoke-Build
            if (-not $success) {
                Write-Host "`nWaiting for fixes... Press ENTER after Claude fixes the errors" -ForegroundColor Yellow
                Read-Host
            }
        } while (-not $success)
        
        Write-Host "`nAll errors fixed! Build successful." -ForegroundColor Green
    }
    default {
        Write-Host "Unknown action: $Action" -ForegroundColor Red
    }
}