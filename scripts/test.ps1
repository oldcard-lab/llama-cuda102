<#
.SYNOPSIS
    Correctness and speed check of a Windows build from scripts/build.ps1, on the Kepler GPU.
.DESCRIPTION
    Runs test-backend-ops against CUDA0, strict per-row mmvk checks, and an optional model comparison/benchmark.
    Also fails if the Windows System log records an NVIDIA driver fault or Display TDR during this run.
.PARAMETER WorkDir
    Parent directory containing llama.cpp (default: work).
.PARAMETER CudaDir
    CUDA 10.2 toolkit directory (default: C:\cuda-10.2).
.EXAMPLE
    .\scripts\test.ps1 work C:\cuda-10.2
#>
param(
    [string] $WorkDir = 'work',
    [string] $CudaDir = 'C:\cuda-10.2'
)

$ErrorActionPreference = 'Continue'
$startedAt = Get-Date
$failed = $false
$originalPath = $env:Path
$originalMmvkNcols = $env:MMVK_CHECK_NCOLS
$originalMmvkTypes = $env:MMVK_CHECK_TYPES

if ([string]::IsNullOrWhiteSpace($env:LLVM_DIR)) {
    $llvmDir = 'C:\Program Files\LLVM\bin'
} else {
    $llvmDir = $env:LLVM_DIR
}

try {
    $workPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkDir)
    $cudaPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CudaDir)
    $llvmDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($llvmDir)
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
    if (-not [string]::IsNullOrEmpty($env:FULL)) { $ops = @('ALL') }

    if (-not (Test-Path -LiteralPath $tbo -PathType Leaf)) {
        Write-Output "missing executable: $tbo"
        $failed = $true
    } else {
        $esc = [char]27
        foreach ($op in $ops) {
            $tboArgs = @('-b', 'CUDA0')
            if ($op -ne 'ALL') { $tboArgs += @('-o', $op) }
            $raw = @(& $tbo @tboArgs 2>&1)
            $tboExit = $LASTEXITCODE
            $out = @($raw | ForEach-Object { ([string]$_ -replace "$esc\[[0-9;]*m", '').TrimEnd() })
            $ok = @($out | Where-Object { $_ -cmatch ': OK$' }).Count
            $bad = @($out | Where-Object { $_ -cmatch ': FAIL$' -and $_ -cnotmatch 'failed to allocate' })
            $big = @($out | Where-Object { $_ -match 'failed to allocate|out of memory' }).Count
            Write-Output ("{0}: OK={1} FAIL={2} (cases too big for this card's memory: {3})" -f $op, $ok, $bad.Count, $big)
            if ($bad.Count -gt 0) {
                $failed = $true
                $bad | Select-Object -First 5 | ForEach-Object { Write-Output $_ }
            }
            if (($out -join "`n") -cmatch 'GGML_ASSERT|Segmentation|Aborted') {
                $failed = $true
                Write-Output "  $op run aborted"
            }
            # Windows-only crash gate: NTSTATUS error exits cover crashes that do not print Unix abort text.
            $tboStatus = [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$tboExit), 0)
            if (($tboStatus -band [uint32]0xC0000000L) -eq [uint32]0xC0000000L) {
                $failed = $true
                Write-Output ("  Windows test binary crash (NTSTATUS 0x{0:X8})" -f $tboStatus)
            }
        }
    }
    Write-Output '   (a tiny q4_1 / q5_1 MUL_MAT case occasionally fails by a hair with random inputs on any build; rerun it)'

    Write-Output '== 2. mmvk-check (strict Kepler kernel check)'
    $libraryNames = @('ggml.lib', 'ggml-base.lib', 'ggml-cuda.lib')
    $libraryDirs = @(
        $binDir,
        (Join-Path $buildDir 'lib'),
        (Join-Path $buildDir 'ggml\src'),
        (Join-Path $buildDir 'ggml\src\ggml-cuda')
    )
    $resolvedLibraries = @{}
    $librariesResolved = $true
    foreach ($libraryName in $libraryNames) {
        $resolvedPath = $null
        $libraryAmbiguous = $false
        foreach ($candidateDir in $libraryDirs) {
            $candidatePath = Join-Path $candidateDir $libraryName
            if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
                $resolvedPath = (Get-Item -LiteralPath $candidatePath).FullName
                break
            }
        }

        if ($null -eq $resolvedPath) {
            $libraryMatches = @(Get-ChildItem -LiteralPath $buildDir -Recurse -File -Filter $libraryName -ErrorAction SilentlyContinue)
            if ($libraryMatches.Count -gt 1) {
                Write-Output ("multiple matches for $libraryName under ${buildDir}:`n  " + ($libraryMatches.FullName -join "`n  "))
                $librariesResolved = $false
                $libraryAmbiguous = $true
            } elseif ($libraryMatches.Count -eq 1) {
                $resolvedPath = $libraryMatches[0].FullName
            }
        }

        if ($null -eq $resolvedPath -and -not $libraryAmbiguous) {
            Write-Output "missing library $libraryName under $buildDir"
            $librariesResolved = $false
        } elseif ($null -ne $resolvedPath) {
            $resolvedLibraries[$libraryName] = $resolvedPath
        }
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
    } elseif (-not $librariesResolved) {
        $failed = $true
    } else {
        $checkerExe = Join-Path $binDir 'mmvk-check.exe'
        $includeDir = Join-Path $src 'ggml\include'
        $compileArgs = @('-O2', '-std=c++17', '--target=x86_64-pc-windows-msvc', '-fms-runtime-lib=dll',
            '-DGGML_SHARED', '-DGGML_BACKEND_SHARED', "-I$includeDir", $checkerSource,
            $resolvedLibraries['ggml.lib'], $resolvedLibraries['ggml-base.lib'],
            $resolvedLibraries['ggml-cuda.lib'], '-fuse-ld=lld', '-o', $checkerExe)
        & $clang @compileArgs
        $compileExit = $LASTEXITCODE
        if ($compileExit -ne 0) {
            Write-Output 'mmvk-check build failed'
            exit 1
        } else {
            $mmvkText = Get-Content -LiteralPath $mmvkSource -Raw -ErrorAction SilentlyContinue
            $ncolsMatch = [regex]::Match([string]$mmvkText, 'MMVK_NCOLS_TYPES:\s*([A-Za-z0-9_,]+)')
            if (-not $ncolsMatch.Success) {
                Write-Output "could not find MMVK_NCOLS_TYPES in $mmvkSource"
                $failed = $true
            } else {
                $env:MMVK_CHECK_NCOLS = $ncolsMatch.Groups[1].Value
                $env:MMVK_CHECK_TYPES = 'q4_K,q6_K,q8_0,q5_0,f16,q3_K,iq3_xxs,q5_K'
                foreach ($mode in @('', '--weights')) {
                    $checkerArgs = @()
                    if ($mode) { $checkerArgs += $mode }
                    $raw = @(& $checkerExe @checkerArgs 2>&1)
                    $checkerExit = $LASTEXITCODE
                    # Windows-only crash gate: report NTSTATUS crashes that have no Unix abort text.
                    $checkerStatus = [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$checkerExit), 0)
                    if (($checkerStatus -band [uint32]0xC0000000L) -eq [uint32]0xC0000000L) {
                        $failed = $true
                        Write-Output ("  Windows test binary crash (NTSTATUS 0x{0:X8})" -f $checkerStatus)
                    }
                    $out = @($raw | ForEach-Object { [string]$_ })
                    $summary = @($out | Where-Object { $_ -cmatch '^SUMMARY' })
                    if ($summary.Count -gt 0) { $summary | ForEach-Object { Write-Output $_ } }
                    else { Write-Output '  no SUMMARY line (crash?)' }
                    @($out | Where-Object { $_ -cmatch '^FAIL' } | Select-Object -First 5) | ForEach-Object { Write-Output $_ }
                    if ($checkerExit -ne 0 -or -not ($out | Where-Object { $_ -cmatch '^SUMMARY.* 0 FAIL' })) { $failed = $true }
                }
            }
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($env:MODEL)) {
        Write-Output "== 3. model: $env:MODEL"
        $prompt = 'The three primary colors are'
        $model = $env:MODEL
        $gpuArgs = @('-m', $model, '-p', $prompt, '-n', '32', '--temp', '0', '-c', '1024', '-ub', '128', '-ngl', '99', '-fa', 'off', '-no-cnv')
        $cpuArgs = @('-m', $model, '-p', $prompt, '-n', '32', '--temp', '0', '-c', '1024', '-ngl', '0', '-fa', 'off', '-no-cnv')
        $completion = Join-Path $binDir 'llama-completion.exe'
        if (-not (Test-Path -LiteralPath $completion -PathType Leaf)) {
            Write-Output "missing executable: $completion"
            $failed = $true
        } else {
            $gpuText = (@(& $completion @gpuArgs 2>$null) | ForEach-Object { [string]$_ }) -join ' '
            $cpuText = (@(& $completion @cpuArgs 2>$null) | ForEach-Object { [string]$_ }) -join ' '
            Write-Output "gpu: $gpuText"
            Write-Output "cpu: $cpuText"
            $gpuRemainder = ([regex]::new([regex]::Escape($prompt))).Replace($gpuText, '', 1).Replace(' ', '')
            if ([string]::IsNullOrWhiteSpace($gpuRemainder)) {
                Write-Output '  no GPU text (does the model fit? try a smaller -c / -ub)'
                $failed = $true
            }
        }

        $bench = Join-Path $binDir 'llama-bench.exe'
        if (-not (Test-Path -LiteralPath $bench -PathType Leaf)) {
            Write-Output "missing executable: $bench"
            $failed = $true
        } else {
            $benchArgs = @('-m', $model, '-ngl', '99', '-fa', '0', '-ub', '128', '-p', '128', '-n', '64', '-r', '2')
            $benchOut = @(& $bench @benchArgs 2>&1)
            $benchLines = @($benchOut | ForEach-Object { [string]$_ } | Where-Object { $_ -cmatch 'pp128|tg64' })
            $benchLines | ForEach-Object { Write-Output $_ }
        }
    }

    Write-Output '== 4. GPU fault check'
    # Windows-only addition: preserve the nvlddmkm-any-event and Display 4101 rule.
    try {
        $events = @(Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $startedAt } -ErrorAction Stop |
            Where-Object { $_.ProviderName -match 'nvlddmkm' -or ($_.ProviderName -eq 'Display' -and $_.Id -eq 4101) })
        $gpuFaults = $events.Count
    } catch {
        if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') { $gpuFaults = 0 }
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
} finally {
    $env:Path = $originalPath
    if ($null -eq $originalMmvkNcols) { Remove-Item Env:\MMVK_CHECK_NCOLS -ErrorAction SilentlyContinue } else { $env:MMVK_CHECK_NCOLS = $originalMmvkNcols }
    if ($null -eq $originalMmvkTypes) { Remove-Item Env:\MMVK_CHECK_TYPES -ErrorAction SilentlyContinue } else { $env:MMVK_CHECK_TYPES = $originalMmvkTypes }
}

if ($failed) {
    Write-Output 'SOME CHECKS FAILED'
    exit 1
}
Write-Output 'ALL CHECKS PASSED'
exit 0
