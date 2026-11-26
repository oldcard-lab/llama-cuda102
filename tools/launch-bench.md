# CUDA launch and submission microbenchmark

This standalone benchmark measures empty-kernel launch overhead, launch-plus-wait latency, a 300-kernel decode-like
shape, periodic stream-query flush probes, and CUDA graph launch cost. It has no llama.cpp dependency. The 300-kernel
shape uses one kernel per simulated token, and each kernel reads and writes a 4 KiB device buffer. Run the same
binary build and GPU workload on Linux and Windows for a direct comparison.

## Build

Linux, from the repository root (set `CUDA` to the CUDA 10.2 toolkit directory):

```bash
clang++-22 -x cuda tools/launch-bench.cu --cuda-gpu-arch=sm_30 --cuda-path="$CUDA" -O2 -L"$CUDA/lib64" -lcudart -o launch-bench
```

Windows PowerShell, from the repository root, after creating the space-free `C:\cuda-10.2` junction described in the
Windows build notes:

```powershell
& 'C:\Program Files\LLVM\bin\clang++.exe' -x cuda tools/launch-bench.cu --cuda-gpu-arch=sm_30 --cuda-path=C:/cuda-10.2 -O2 -fuse-ld=lld -LC:/cuda-10.2/lib/x64 -lcudart -o launch-bench.exe
```

The CUDA runtime DLL must be available at execution time. On Windows, add `C:\cuda-10.2\bin` to `PATH` in the
PowerShell session that runs the program.

## Run

Run each scheduling mode as a separate process so `cudaSetDeviceFlags` is applied before CUDA context creation:

```text
launch-bench [auto|spin|yield|blocking]
```

Omitting the argument selects `auto`. Each measurement is printed as `name value unit`; the first line identifies the
mode. The benchmark is intentionally short-lived and should be run with the same power plan, display/GPU usage, driver,
and toolkit on both operating systems.

## Measurements and interpretation

| Name | Meaning |
|---|---|
| `launch_queue_us` | CPU wall time to enqueue 100,000 empty kernels, measured before synchronization and divided by launch count. Estimates host-side enqueue cost per launch. |
| `drain_us` | Total time from the same queue start through the final stream synchronization, divided by launch count. Includes enqueue and draining the queued work. |
| `roundtrip_us` | Mean time for 2,000 empty-kernel launch plus stream-synchronize iterations. Measures serialized launch/wait latency. |
| `decode_shape_ms` | Mean time per trial for 300 4 KiB-touching kernels followed by one stream synchronization; 200 trials. Approximates a decode token made of many small dependent submissions. |
| `flush_N_ms` | Same 300-kernel shape, calling `cudaStreamQuery` every N launches (N = 1, 4, 16, 64), then synchronizing once at the end of each trial. A lower value at small N is evidence that periodic query/flush changes submission batching behavior. |
| `graph_instantiate_ms` | One-time host elapsed time for instantiating a captured graph containing the same 300 kernels. |
| `graph_ms` | Mean time to launch that graph and synchronize, over 200 trials. Compare with `decode_shape_ms` to estimate whether graphs reduce submission overhead on this driver. |

The analysis motivating this probe found prompt processing parity but about a threefold token-generation gap on the same
Kepler card. That pattern points toward the many small kernel submissions in decode rather than large-kernel GPU
throughput. A Windows `launch_queue_us` increase quantifies host enqueue cost; a much larger `roundtrip_us` or decode
shape gap suggests submission or wait latency. If graph launch is substantially faster than the uncaptured shape,
graphs are a promising follow-up, but this microbenchmark alone does not prove the root cause in llama.cpp. The
separate `flush_N_ms` values test whether periodic stream queries alter the batching behavior. Scheduling-mode
differences can reveal whether host wait policy is material, but do not explain the full per-kernel path by themselves.

If CUDA graph capture or instantiation is unsupported or fails on the installed runtime/driver, the program prints
`graph_error` with the CUDA error text and continues with normal cleanup.

## Results

| Measurement | Linux | Windows |
|---|---:|---:|
| `launch_queue_us` |  |  |
| `drain_us` |  |  |
| `roundtrip_us` |  |  |
| `decode_shape_ms` |  |  |
| `flush_1_ms` |  |  |
| `flush_4_ms` |  |  |
| `flush_16_ms` |  |  |
| `flush_64_ms` |  |  |
| `graph_instantiate_ms` |  |  |
| `graph_ms` |  |  |
