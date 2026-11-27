# Install the CUDA 10.2 toolkit on Windows for this build (toolkit subpackages only; your current NVIDIA driver stays).
# CUDA 10.2 is the last toolkit for compute capability 3.0 (sm_30). Run in an elevated PowerShell:
#   scripts\install-cuda-10.2.ps1 [DOWNLOAD_DIR] [LINK]
#   DOWNLOAD_DIR  where the 2.6 GB installer is kept (default .\cuda-installer)
#   LINK          space-free path to the toolkit, created as a directory junction (default C:\cuda-10.2). CMake splits
#                 a --cuda-path at a space, so build.ps1 uses this path, not "C:\Program Files\...".
# md5 from NVIDIA's md5sum.txt for 10.2; the installer's Authenticode signer must be NVIDIA Corporation.
param([string]$DownloadDir = '.\cuda-installer', [string]$Link = 'C:\cuda-10.2')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$DownloadDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DownloadDir)
$Link = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Link)
$name = 'cuda_10.2.89_441.22_win10.exe'
$url  = "https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/$name"
$md5  = 'd9f5b9f24c3d3fc456a3c789f9b43419'
$root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2'

if ($Link -match '\s') {
    Write-Output 'LINK must not contain spaces; use a space-free junction path.'
    exit 2
}

$id = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $id.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Output 'Run this from an elevated PowerShell (Run as administrator).'
    exit 2
}

New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
$exe = Join-Path $DownloadDir $name
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Output "fetch $url (2.6 GB)"
    $partial = "$exe.part"
    Invoke-WebRequest -Uri $url -OutFile $partial -UseBasicParsing
    Move-Item -LiteralPath $partial -Destination $exe
}
$h = (Get-FileHash -LiteralPath $exe -Algorithm MD5).Hash.ToLower()
if ($h -ne $md5) {
    Write-Output "installer file is bad: md5 $h, expected $md5; deleting it."
    Remove-Item -LiteralPath $exe -Force
    exit 3
}
Write-Output "md5 ok $h"

$sig = Get-AuthenticodeSignature -LiteralPath $exe
if ($sig.Status -ne 'Valid') {
    Write-Output "signature check failed: $($sig.Status)"
    exit 3
}
$subject = $sig.SignerCertificate.Subject
Write-Output "signature $($sig.Status) / $subject"
if ($subject -notmatch '(^|,\s*)O=NVIDIA Corporation(,|$)') {
    Write-Output 'signature check failed: signer is not NVIDIA Corporation.'
    exit 3
}

# clang's CUDA wrapper header includes curand_mtgp32_kernel.h, so curand + curand_dev are needed too.
$sub = 'nvcc_10.2', 'cudart_10.2', 'cublas_10.2', 'cublas_dev_10.2', 'curand_10.2', 'curand_dev_10.2',
       'cuobjdump_10.2', 'nvdisasm_10.2', 'nvprune_10.2', 'nvrtc_10.2', 'nvrtc_dev_10.2', 'nvml_dev_10.2'
Write-Output "setup -s $($sub -join ' ')"
$p = Start-Process -FilePath $exe -ArgumentList (@('-s') + $sub) -Wait -PassThru
Write-Output "setup exit $($p.ExitCode)"
if ($p.ExitCode -eq 3010) {
    Write-Output 'reboot recommended'
} elseif ($p.ExitCode -ne 0) {
    Write-Output "CUDA installer failed with exit code $($p.ExitCode)."
    exit 4
}

if (Test-Path -LiteralPath $Link) {
    $linkItem = Get-Item -LiteralPath $Link -Force
    if (-not ($linkItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        Write-Output "$Link exists and is not a junction to $root."
        exit 4
    }
    $reparseInfo = & fsutil.exe reparsepoint query $Link
    if ($LASTEXITCODE -ne 0) {
        Write-Output "could not verify the existing junction at $Link."
        exit 4
    }
    $targetMatch = [regex]::Match(($reparseInfo -join "`n"), '(?m)^\s*Substitute Name:\s*(.+?)\s*$')
    if (-not $targetMatch.Success) {
        Write-Output "could not read the junction target at $Link."
        exit 4
    }
    $existingTarget = $targetMatch.Groups[1].Value -replace '^\\\?\?\\', ''
    if ($existingTarget.TrimEnd('\') -ine $root.TrimEnd('\')) {
        Write-Output "$Link exists and points to $existingTarget, not $root."
        exit 4
    }
    Write-Output "$Link exists; not changing it"
} else {
    New-Item -ItemType Junction -Path $Link -Target $root | Out-Null
}

$versionFile = Join-Path $Link 'version.txt'
$version = Get-Content -LiteralPath $versionFile -Raw -ErrorAction SilentlyContinue
if ($version -notmatch 'CUDA Version 10\.2') {
    Write-Output "CUDA 10.2 not found at $Link."
    exit 4
}
$requiredFiles = @(
    'bin\nvcc.exe', 'bin\cudart64_102.dll', 'bin\cublas64_10.dll',
    'lib\x64\cudart.lib', 'lib\x64\cublas.lib', 'include\curand_mtgp32_kernel.h'
)
$missingFiles = @($requiredFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Link $_) -PathType Leaf) })
if ($missingFiles.Count -gt 0) {
    Write-Output "CUDA toolkit files missing at ${Link}: $($missingFiles -join ', ')"
    exit 4
}
Write-Output "CUDA 10.2 ready at $Link ($($version.Trim()))"
Write-Output "display driver: $((Get-CimInstance Win32_VideoController | Where-Object Name -match NVIDIA).DriverVersion)"
