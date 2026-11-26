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
$name = 'cuda_10.2.89_441.22_win10.exe'
$url  = "https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/$name"
$md5  = 'd9f5b9f24c3d3fc456a3c789f9b43419'
$root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2'

$id = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $id.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Run this from an elevated PowerShell (Run as administrator).'; exit 2
}

New-Item -ItemType Directory -Force $DownloadDir | Out-Null
$exe = Join-Path (Resolve-Path $DownloadDir) $name
if (-not (Test-Path $exe)) {
    Write-Host "fetch $url (2.6 GB)"
    Invoke-WebRequest -Uri $url -OutFile "$exe.part" -UseBasicParsing
    Move-Item "$exe.part" $exe
}
$h = (Get-FileHash $exe -Algorithm MD5).Hash.ToLower()
if ($h -ne $md5) { Write-Host "md5 mismatch: $h, expected $md5"; exit 3 }
Write-Host "md5 ok $h"
$sig = Get-AuthenticodeSignature $exe
Write-Host "signature $($sig.Status) / $($sig.SignerCertificate.Subject)"
if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'O=NVIDIA Corporation') { Write-Host 'signature check failed'; exit 3 }

# clang's CUDA wrapper header includes curand_mtgp32_kernel.h, so curand + curand_dev are needed too.
$sub = 'nvcc_10.2', 'cudart_10.2', 'cublas_10.2', 'cublas_dev_10.2', 'curand_10.2', 'curand_dev_10.2',
       'cuobjdump_10.2', 'nvdisasm_10.2', 'nvprune_10.2', 'nvrtc_10.2', 'nvrtc_dev_10.2', 'nvml_dev_10.2'
Write-Host "setup -s $($sub -join ' ')"
$p = Start-Process -FilePath $exe -ArgumentList (@('-s') + $sub) -Wait -PassThru
Write-Host "setup exit $($p.ExitCode)"

if (-not (Test-Path $Link)) { cmd /c mklink /J $Link "$root" | Out-Host }
$ver = Get-Content (Join-Path $Link 'version.txt') -ErrorAction SilentlyContinue
if ("$ver" -notmatch 'CUDA Version 10\.2') { Write-Host "CUDA 10.2 not found at $Link"; exit 4 }
if (-not (Test-Path (Join-Path $Link 'include\curand_mtgp32_kernel.h'))) { Write-Host 'curand headers missing'; exit 4 }
Write-Host "CUDA 10.2 ready at $Link ($ver)"
Write-Host "display driver: $((Get-CimInstance Win32_VideoController | Where-Object Name -match NVIDIA).DriverVersion)"
