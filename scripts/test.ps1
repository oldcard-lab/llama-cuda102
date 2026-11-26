<#
.SYNOPSIS
    Correctness and speed check of a build from scripts/build.sh, on the Kepler GPU.
.DESCRIPTION
    Runs test-backend-ops against CUDA0, strict per-row mmvk checks, and an optional model comparison/benchmark.
    Also fails if the Windows System log records an NVIDIA driver fault or Display TDR during this run.
.PARAMETER WorkDir
    Parent directory containing llama.cpp (default: work).
.PARAMETER CudaDir
    CUDA 10.2 toolkit directory (default: $HOME/cuda-10.2).
.EXAMPLE
    .\scripts\test.ps1 work C:\cuda-10.2
#>
param(
    [string] $WorkDir = 'work',
    [string] $CudaDir = (Join-Path $HOME 'cuda-10.2')
)

$ErrorActionPreference = 'Continue'
$startedAt = Get-Date
$failed = $false

if ([string]::IsNullOrWhiteSpace($env:LLVM_DIR)) {
    $llvmDir = 'C:\Program Files\LLVM\bin'
} else {
    $llvmDir = $env:LLVM_DIR
}

try {
    $workPath = [System.IO.Path]::GetFullPath($WorkDir)
    $cudaPath = [System.IO.Path]::GetFullPath($CudaDir)
    $src = Join-Path $workPath 'llama.cpp'
    $buildDir = Join-Path $src 'build'
    $binDir = Join-Path $buildDir 'bin'
    $tbo = Join-Path $binDir 'test-backend-ops.exe'
    $clang = Join-Path $llvmDir 'clang++.exe'
    $checkerSource = Join-Path (Split-Path -Parent $PSScriptRoot) 'tools\mmvk-check.cpp'
    $mmvkSource = Join-Path $src 'ggml\src\ggml-cuda\mmvk.cu'

    # Keep the caller's PATH and put the CUDA runtime first for all child processes.
    $env:Path = (Join-Path $cudaPath 'bin') + ';' + $env:Path

    Write-Output '== 1. test-backend-ops (CUDA0 vs CPU)'
    $ops = @('MUL_MAT', 'MUL_MAT_ID', 'IM2COL', 'CONV_2D')
    if ($env:FULL -eq '1') { $ops = @('ALL') }

    if (-not (Test-Path -LiteralPath $tbo -PathType Leaf)) {
        Write-Output "missing executable: $tbo"
        $failed = $true
    } else {
        $esc = [char]27
        foreach ($op in $ops) {
            $args = @('-b', 'CUDA0')
            if ($op -ne 'ALL') { $args += @('-o', $op) }
            $raw = @(& $tbo @args 2>&1)
            $tboExit = $LASTEXITCODE
            $out = @($raw | ForEach-Object { ([string]$_ -replace "$esc\[[0-9;]*m", '').TrimEnd() })
            $ok = @($out | Where-Object { $_ -match ': OK$' }).Count
            $bad = @($out | Where-Object { $_ -match ': FAIL$' -and $_ -notmatch 'failed to allocate' })
            $big = @($out | Where-Object { $_ -match 'failed to allocate|out of memory' }).Count
            Write-Output ("{0}: OK={1} FAIL={2} (cases too big for this card's memory: {3})" -f $op, $ok, $bad.Count, $big)
            if ($bad.Count -gt 0 -or $tboExit -ne 0) {
                $failed = $true
                $bad | Select-Object -First 5 | ForEach-Object { Write-Output $_ }
            }
            if (($out -join "`n") -match 'GGML_ASSERT|Segmentation|Aborted|exception') {
                $failed = $true
                Write-Output "  $op run aborted"
            }
        }
    }
    Write-Output '   (a tiny q4_1 / q5_1 MUL_MAT case occasionally fails by a hair with random inputs on any build; rerun it)'

    Write-Output '== 2. mmvk-check (strict Kepler kernel check)'
    $libs = @{}
    foreach ($name in @('ggml.lib', 'ggml-base.lib', 'ggml-cuda.lib')) {
        $match = Get-ChildItem -LiteralPath $buildDir -Filter $name -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $match) { $libs[$name] = $match.FullName }
    }

    if (-not (Test-Path -LiteralPath $clang -PathType Leaf)) {
        Write-Output "missing compiler: $clang"
        $failed = $true
    } elseif (-not (Test-Path -LiteralPath $checkerSource -PathType Leaf)) {
        Write-Output "missing source: $checkerSource"
        $failed = $true
    } elseif (-not (Test-Path -LiteralPath (Join-Path $src 'ggml\include\ggml.h') -PathType Leaf)) {
        Write-Output "missing ggml headers under: $src"
        $failed = $true
    } elseif ($libs.Count -ne 3) {
        Write-Output "mmvk-check build failed: could not find ggml.lib, ggml-base.lib, and ggml-cuda.lib under $buildDir"
        $failed = $true
    } else {
        $checkerExe = Join-Path $binDir 'mmvk-check.exe'
        $includeDir = Join-Path $src 'ggml\include'
        $compileArgs = @('-O2', '-std=c++17', '--target=x86_64-pc-windows-msvc', "-I$includeDir", $checkerSource,
            $libs['ggml.lib'], $libs['ggml-base.lib'], $libs['ggml-cuda.lib'], '-fuse-ld=lld', '-o', $checkerExe)
        & $clang @compileArgs
        $compileExit = $LASTEXITCODE
        if ($compileExit -ne 0) {
            Write-Output 'mmvk-check build failed'
            $failed = $true
        } else {
            $mmvkText = Get-Content -LiteralPath $mmvkSource -Raw -ErrorAction SilentlyContinue
            $ncolsMatch = [regex]::Match([string]$mmvkText, 'MMVK_NCOLS_TYPES:\s*([A-Za-z0-9_,]+)')
            if (-not $ncolsMatch.Success) {
                Write-Output "could not find MMVK_NCOLS_TYPES in $mmvkSource"
                $failed = $true
            } else {
                $oldNcols = $env:MMVK_CHECK_NCOLS
                $oldTypes = $env:MMVK_CHECK_TYPES
                $env:MMVK_CHECK_NCOLS = $ncolsMatch.Groups[1].Value
                $env:MMVK_CHECK_TYPES = 'q4_K,q6_K,q8_0,q5_0,f16,q3_K,iq3_xxs,q5_K'
                foreach ($mode in @('', '--weights')) {
                    $checkerArgs = @()
                    if ($mode) { $checkerArgs += $mode }
                    $raw = @(& $checkerExe @checkerArgs 2>&1)
                    $checkerExit = $LASTEXITCODE
                    $out = @($raw | ForEach-Object { [string]$_ })
                    $summary = @($out | Where-Object { $_ -match '^SUMMARY' })
                    if ($summary.Count -gt 0) { $summary | ForEach-Object { Write-Output $_ } }
                    else { Write-Output '  no SUMMARY line (crash?)' }
                    @($out | Where-Object { $_ -match '^FAIL' } | Select-Object -First 5) | ForEach-Object { Write-Output $_ }
                    if ($checkerExit -ne 0 -or -not ($out | Where-Object { $_ -match '^SUMMARY.* 0 FAIL' })) { $failed = $true }
                }
                if ($null -eq $oldNcols) { Remove-Item Env:\MMVK_CHECK_NCOLS -ErrorAction SilentlyContinue } else { $env:MMVK_CHECK_NCOLS = $oldNcols }
                if ($null -eq $oldTypes) { Remove-Item Env:\MMVK_CHECK_TYPES -ErrorAction SilentlyContinue } else { $env:MMVK_CHECK_TYPES = $oldTypes }
            }
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($env:MODEL)) {
        Write-Output "== 3. model: $env:MODEL"
        $prompt = 'The three primary colors are'
        $model = $env:MODEL
        $gpuArgs = @('-m', $model, '-p', $prompt, '-n', '32', '--temp', '0', '-c', '1024', '-ub', '128', '-ngl', '99', '-fa', 'off', '-no-cnv')
        $cpuArgs = @('-m', $model, '-p', $prompt, '-n', '32', '--temp', '0', '-c', '1024', '-ngl', '0', '-fa', 'off', '-no-cnv')
        $gpuText = (@(& (Join-Path $binDir 'llama-completion.exe') @gpuArgs 2>$null) | ForEach-Object { [string]$_ }) -join ' '
        $gpuExit = $LASTEXITCODE
        $cpuText = (@(& (Join-Path $binDir 'llama-completion.exe') @cpuArgs 2>$null) | ForEach-Object { [string]$_ }) -join ' '
        $cpuExit = $LASTEXITCODE
        Write-Output "gpu: $gpuText"
        Write-Output "cpu: $cpuText"
        $gpuRemainder = $gpuText.Replace($prompt, '').Replace(' ', '')
        if ([string]::IsNullOrWhiteSpace($gpuRemainder)) {
            Write-Output '  no GPU text (does the model fit? try a smaller -c / -ub)'
            $failed = $true
        }
        if ($gpuExit -ne 0 -or $cpuExit -ne 0) { $failed = $true }

        $bench = Join-Path $binDir 'llama-bench.exe'
        $benchArgs = @('-m', $model, '-ngl', '99', '-fa', '0', '-ub', '128', '-p', '128', '-n', '64', '-r', '2')
        $benchOut = @(& $bench @benchArgs 2>&1)
        $benchExit = $LASTEXITCODE
        $benchLines = @($benchOut | ForEach-Object { [string]$_ } | Where-Object { $_ -match 'pp128|tg64' })
        $benchLines | ForEach-Object { Write-Output $_ }
        if ($benchExit -ne 0 -or $benchLines.Count -eq 0) { $failed = $true }
    }

    Write-Output '== 4. GPU fault check'
    try {
        $events = @(Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $startedAt } -ErrorAction Stop |
            Where-Object { $_.ProviderName -match 'nvlddmkm' -or ($_.ProviderName -eq 'Display' -and $_.Id -eq 4101) })
        $gpuFaults = $events.Count
    } catch {
        if ($_.Exception.Message -match 'No events were found') { $gpuFaults = 0 }
        else {
            Write-Output ("GPU fault event query failed: {0}" -f $_.Exception.Message)
            $gpuFaults = -1
            $failed = $true
        }
    }
    Write-Output "GPU fault events: $gpuFaults"
    if ($gpuFaults -gt 0) { $failed = $true }
} catch {
    Write-Output ("test script error: {0}" -f $_.Exception.Message)
    $failed = $true
}

if ($failed) {
    Write-Output 'SOME CHECKS FAILED'
    exit 1
}
Write-Output 'ALL CHECKS PASSED'
exit 0
