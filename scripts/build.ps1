# Fetch llama.cpp v1.0.0, apply the patch series, and build the CUDA backend for Kepler (sm_30).
# Usage: scripts/build.ps1 [WORK_DIR] [CUDA_DIR]
# Env: TARGETS, JOBS, LLVM_DIR (default: C:\Program Files\LLVM\bin).
param(
    [string]$WorkDir = '.\work',
    [string]$CudaDir = 'C:\cuda-10.2'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$work = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkDir)
$CudaDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CudaDir)
$llvm = if ($env:LLVM_DIR) { $env:LLVM_DIR } else { 'C:\Program Files\LLVM\bin' }
$llvm = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($llvm)
$src = Join-Path $work 'llama.cpp'
$baseCommit = 'b1b770d90d94cad9e9ea5b993bdbc383a522fb28'
$upstream = 'https://github.com/ggml-org/llama.cpp.git'
$targets = if ($env:TARGETS) { $env:TARGETS } else { 'llama-cli llama-server llama-completion llama-bench test-backend-ops' }
$clangMajor = if ($env:CLANG_VERSION) { $env:CLANG_VERSION } else { '22' }
$jobs = if ($env:JOBS) { $env:JOBS } else { [Environment]::ProcessorCount }

foreach ($tool in @('git', 'cmake', 'ninja')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Output "missing tool: $tool"
        exit 2
    }
}
foreach ($compiler in @('clang.exe', 'clang++.exe', 'llvm-rc.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $llvm $compiler))) {
        Write-Output "missing tool: $(Join-Path $llvm $compiler)"
        exit 2
    }
}
$lldLink = Join-Path $llvm 'lld-link.exe'
if (-not (Test-Path -LiteralPath $lldLink)) {
    Write-Output "missing tool: $lldLink"
    exit 2
}
$clangVersion = & (Join-Path $llvm 'clang.exe') --version
if ($LASTEXITCODE -ne 0) { Write-Output "clang version check failed ($LASTEXITCODE)"; exit 2 }
if (($clangVersion -join "`n") -notmatch "(?m)\bversion $([regex]::Escape($clangMajor))(?:\.|\s)") {
    Write-Output "LLVM clang major version $clangMajor is required."
    exit 2
}

if ($CudaDir -match '\s') {
    Write-Output "CUDA_DIR must not contain spaces; create the junction with scripts/install-cuda-10.2.ps1."
    exit 2
}
$versionFile = Join-Path $CudaDir 'version.txt'
if (-not (Test-Path -LiteralPath $versionFile) -or ((Get-Content -LiteralPath $versionFile -Raw) -notmatch 'CUDA Version 10\.2')) {
    Write-Output "CUDA 10.2 toolkit not found in $CudaDir (scripts/install-cuda-10.2.ps1)"
    exit 2
}

New-Item -ItemType Directory -Force -Path $work | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $src '.git'))) {
    & git -c core.autocrlf=false clone --branch v1.0.0 --depth 1 $upstream $src
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& git -C $src config core.autocrlf false
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$oldErrorActionPreference = $ErrorActionPreference
$nativePreference = Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue
if ($nativePreference) { $oldNativePreference = $PSNativeCommandUseErrorActionPreference }
try {
    $ErrorActionPreference = 'Continue'
    if ($nativePreference) { $PSNativeCommandUseErrorActionPreference = $false }
    & git -C $src merge-base --is-ancestor $baseCommit HEAD 2>$null
    $ancestorExit = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $oldErrorActionPreference
    if ($nativePreference) { $PSNativeCommandUseErrorActionPreference = $oldNativePreference }
}
if ($ancestorExit -ne 0) {
    Write-Output "$src is not llama.cpp v1.0.0 ($baseCommit); it may be from an older version of this repository."
    Write-Output "Delete $src and run again."
    exit 3
}

$patchFiles = @(Get-ChildItem -LiteralPath (Join-Path $repo 'patches') -Filter '*.patch' -File | Sort-Object Name)
if ($patchFiles.Count -eq 0) {
    Write-Output 'patches directory contains no .patch files.'
    exit 4
}
$sha = [System.Security.Cryptography.SHA256]::Create()
try {
    foreach ($patch in $patchFiles) {
        $bytes = [System.IO.File]::ReadAllBytes($patch.FullName)
        [void]$sha.TransformBlock($bytes, 0, $bytes.Length, $bytes, 0)
    }
    [void]$sha.TransformFinalBlock([byte[]]@(), 0, 0)
    $patchSum = ([BitConverter]::ToString($sha.Hash)).Replace('-', '').ToLowerInvariant()
}
finally { $sha.Dispose() }

$subjects = & git -C $src log --format=%s
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$marker = Join-Path $src '.git\llama-cuda102-series'
$hasSeries = (($subjects -join "`n") -match '6uO, llama\.cpp discussion #29961') -and
             (($subjects -join "`n") -match 'FunctionGemma tool-call parser')
if ($hasSeries) {
    $markedSum = if (Test-Path -LiteralPath $marker) { ([string](Get-Content -LiteralPath $marker -Raw)).Trim() } else { '' }
    if ($markedSum -ne $patchSum) {
        Write-Output "$src carries a different version of this patch series. Delete $src and run again."
        exit 4
    }
} else {
    $head = & git -C $src rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($head -eq $baseCommit) {
        $patchPaths = @($patchFiles | ForEach-Object { $_.FullName })
        & git -C $src -c user.name=oldcard-lab -c user.email=oldcard-lab@localhost am --keep-cr @patchPaths
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        [System.IO.File]::WriteAllText($marker, $patchSum + "`n", [System.Text.Encoding]::ASCII)
    } else {
        Write-Output "$src has other commits on top of v1.0.0 (an older or partly applied series). Delete it and run again."
        exit 4
    }
}

$cudaForward = $CudaDir -replace '\\', '/'
$cmakeArgs = @(
    '-S', $src, '-B', (Join-Path $src 'build'), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    "-DCMAKE_C_COMPILER=$(Join-Path $llvm 'clang.exe')",
    "-DCMAKE_CXX_COMPILER=$(Join-Path $llvm 'clang++.exe')",
    "-DCMAKE_CUDA_COMPILER=$(Join-Path $llvm 'clang++.exe')",
    "-DCMAKE_CUDA_HOST_COMPILER=$(Join-Path $llvm 'clang++.exe')",
    # Windows only: clang's GNU-style driver enables the RC language; LLVM is not on PATH, so name llvm-rc directly.
    "-DCMAKE_RC_COMPILER=$((Join-Path $llvm 'llvm-rc.exe') -replace '\\', '/')",
    "-DCUDAToolkit_ROOT=$cudaForward", '-DCMAKE_CUDA_ARCHITECTURES=30',
    "-DCMAKE_CUDA_FLAGS=--cuda-path=$cudaForward --no-cuda-version-check -Wno-unknown-cuda-version",
    '-DCMAKE_LINKER_TYPE=LLD', '-DGGML_CUDA=ON', '-DGGML_NATIVE=ON',
    '-DGGML_CUDA_GRAPHS=OFF', '-DGGML_CUDA_FA=OFF', '-DGGML_CUDA_NCCL=OFF', '-DGGML_CUDA_NO_VMM=ON',
    '-DLLAMA_BUILD_HTML=OFF', '-DLLAMA_USE_PREBUILT_UI=OFF', '-DLLAMA_CURL=OFF',
    '-DLLAMA_BUILD_TESTS=ON', '-DLLAMA_BUILD_SERVER=ON', '-DLLAMA_BUILD_TOOLS=ON'
)
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$targetList = @($targets -split '\s+' | Where-Object { $_ })
& cmake --build (Join-Path $src 'build') -j $jobs --target @targetList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "built: $(Join-Path $src 'build\bin')"
