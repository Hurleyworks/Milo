# Claude Code Test Integration Command for Unit Tests
# This script builds and runs unit tests, capturing all output for Claude

param(
    [string]$TestName = ""  # Specific test to run, or empty for all tests
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
        Write-Host "BUILD SUCCEEDED!" -ForegroundColor Green
        return $true
    }
    else {
        Write-Host "BUILD FAILED" -ForegroundColor Red
        
        # Format errors for Claude
        if (Test-Path "build_errors.txt") {
            $content = Get-Content "build_errors.txt" -Raw
            $formatted = "=== Unit Test Build Errors ===" + "`n"
            $formatted += "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" + "`n"
            $formatted += "=" * 50 + "`n`n"
            $formatted += $content
            $formatted | Out-File -FilePath "build_errors.txt" -Encoding UTF8
            
            Write-Host "`nBuild errors saved to: unittest/build_errors.txt" -ForegroundColor Yellow
        }
        
        return $false
    }
}

function Run-Tests {
    param([string]$TestFilter)
    
    Write-Host "`nRunning tests..." -ForegroundColor Cyan
    
    # Clean previous test results
    if (Test-Path "test_results.txt") { Remove-Item "test_results.txt" }
    
    $TestExeDir = "..\builds\bin\Debug-windows-x86_64"
    $TestResults = @()
    $FailedTests = @()
    $PassedTests = @()
    
    # Get all test executables
    if ($TestFilter) {
        $TestExes = Get-ChildItem -Path $TestExeDir -Directory | Where-Object { $_.Name -like "*$TestFilter*" }
    } else {
        $TestExes = Get-ChildItem -Path $TestExeDir -Directory
    }
    
    foreach ($TestDir in $TestExes) {
        $ExePath = Join-Path $TestDir.FullName "$($TestDir.Name).exe"
        if (Test-Path $ExePath) {
            Write-Host "Running: $($TestDir.Name)..." -ForegroundColor Gray
            
            # Run test and capture output
            $Output = & $ExePath 2>&1 | Out-String
            
            $TestResults += "`n" + "=" * 70
            $TestResults += "`nTEST: $($TestDir.Name)"
            $TestResults += "`n" + "=" * 70
            $TestResults += "`n$Output"
            
            # Check if test passed
            if ($Output -match "Status: SUCCESS" -or $Output -match "All tests passed") {
                $PassedTests += $TestDir.Name
                Write-Host "  [PASS]" -ForegroundColor Green
            } else {
                $FailedTests += $TestDir.Name
                Write-Host "  [FAIL]" -ForegroundColor Red
            }
        }
    }
    
    # Write results to file
    $Summary = "=== Unit Test Results ===" + "`n"
    $Summary += "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" + "`n"
    $Summary += "=" * 50 + "`n`n"
    
    if ($PassedTests.Count -gt 0) {
        $Summary += "PASSED TESTS ($($PassedTests.Count)):" + "`n"
        $PassedTests | ForEach-Object { $Summary += "  [PASS] $_" + "`n" }
        $Summary += "`n"
    }
    
    if ($FailedTests.Count -gt 0) {
        $Summary += "FAILED TESTS ($($FailedTests.Count)):" + "`n"
        $FailedTests | ForEach-Object { $Summary += "  [FAIL] $_" + "`n" }
        $Summary += "`n"
    }
    
    $Summary += "`nDETAILED OUTPUT:" + "`n"
    $Summary += $TestResults -join "`n"
    
    $Summary | Out-File -FilePath "test_results.txt" -Encoding UTF8
    
    # Display summary
    Write-Host "`n" + "=" * 50 -ForegroundColor Cyan
    Write-Host "TEST SUMMARY" -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host "Passed: $($PassedTests.Count)" -ForegroundColor Green
    $FailedColor = if ($FailedTests.Count -eq 0) { "Green" } else { "Red" }
    Write-Host "Failed: $($FailedTests.Count)" -ForegroundColor $FailedColor
    
    if ($FailedTests.Count -gt 0) {
        Write-Host "`nTest results saved to: unittest/test_results.txt" -ForegroundColor Yellow
        return $false
    }
    
    return $true
}

# Main execution
Write-Host "Unit Test Runner" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Build first
$BuildSuccess = Invoke-Build

if ($BuildSuccess) {
    # Run tests
    $TestSuccess = Run-Tests -TestFilter $TestName
    
    if ($TestSuccess) {
        Write-Host "`nALL TESTS PASSED!" -ForegroundColor Green
    } else {
        Write-Host "`nSOME TESTS FAILED - Check unittest/test_results.txt for details" -ForegroundColor Red
    }
} else {
    Write-Host "`nBuild failed - Check unittest/build_errors.txt for details" -ForegroundColor Red
}