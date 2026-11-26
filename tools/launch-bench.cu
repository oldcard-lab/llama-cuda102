#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>

using clock_type = std::chrono::steady_clock;

__global__ void empty_kernel() {}

__global__ void touch_4k(unsigned int * data) {
    const unsigned int i = threadIdx.x;
    data[i] = data[i] + 1;
}

static bool check(cudaError_t status, const char * operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "error %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

static double elapsed_ms(clock_type::time_point start, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static bool launch_empty(cudaStream_t stream) {
    empty_kernel<<<1, 32, 0, stream>>>();
    return true;
}

static bool launch_shape(cudaStream_t stream, unsigned int * data, int query_every) {
    for (int i = 1; i <= 300; ++i) {
        touch_4k<<<1, 1024, 0, stream>>>(data);
        if (query_every && i % query_every == 0) {
            const cudaError_t status = cudaStreamQuery(stream);
            if (status != cudaSuccess && status != cudaErrorNotReady) {
                return check(status, "cudaStreamQuery");
            }
        }
    }
    return true;
}

static bool run_measurements(cudaStream_t stream, unsigned int * data) {
    const int queue_count = 100000;
    auto t0 = clock_type::now();
    for (int i = 0; i < queue_count; ++i) launch_empty(stream);
    auto t1 = clock_type::now();
    if (!check(cudaStreamSynchronize(stream), "queue drain synchronize")) return false;
    auto t2 = clock_type::now();
    std::printf("launch_queue_us %.6f us\n", elapsed_ms(t0, t1) * 1000.0 / queue_count);
    std::printf("drain_us %.6f us\n", elapsed_ms(t0, t2) * 1000.0 / queue_count);

    const int roundtrip_count = 2000;
    t0 = clock_type::now();
    for (int i = 0; i < roundtrip_count; ++i) {
        launch_empty(stream);
        if (!check(cudaStreamSynchronize(stream), "roundtrip synchronize")) return false;
    }
    t1 = clock_type::now();
    std::printf("roundtrip_us %.6f us\n", elapsed_ms(t0, t1) * 1000.0 / roundtrip_count);

    const int shape_count = 200;
    t0 = clock_type::now();
    for (int i = 0; i < shape_count; ++i) {
        if (!launch_shape(stream, data, 0)) return false;
        if (!check(cudaStreamSynchronize(stream), "decode shape synchronize")) return false;
    }
    t1 = clock_type::now();
    std::printf("decode_shape_ms %.6f ms\n", elapsed_ms(t0, t1) / shape_count);

    const int flush_counts[] = { 1, 4, 16, 64 };
    for (int n : flush_counts) {
        t0 = clock_type::now();
        for (int i = 0; i < shape_count; ++i) {
            if (!launch_shape(stream, data, n)) return false;
            if (!check(cudaStreamSynchronize(stream), "flush probe synchronize")) return false;
        }
        t1 = clock_type::now();
        std::printf("flush_%d_ms %.6f ms\n", n, elapsed_ms(t0, t1) / shape_count);
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaError_t status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (status == cudaSuccess) {
        for (int i = 0; i < 300; ++i) touch_4k<<<1, 1024, 0, stream>>>(data);
        status = cudaStreamEndCapture(stream, &graph);
    }
    const auto instantiate_start = clock_type::now();
    if (status == cudaSuccess) status = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    const auto instantiate_end = clock_type::now();
    if (status != cudaSuccess) {
        std::printf("graph_error %s\n", cudaGetErrorString(status));
        if (graph_exec) check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy after graph error");
        if (graph) check(cudaGraphDestroy(graph), "cudaGraphDestroy after graph error");
        return true;
    }
    std::printf("graph_instantiate_ms %.6f ms\n", elapsed_ms(instantiate_start, instantiate_end));

    t0 = clock_type::now();
    for (int i = 0; i < shape_count; ++i) {
        if (!check(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch")) {
            check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy after launch error");
            check(cudaGraphDestroy(graph), "cudaGraphDestroy after launch error");
            return false;
        }
        if (!check(cudaStreamSynchronize(stream), "graph synchronize")) {
            check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy after synchronize error");
            check(cudaGraphDestroy(graph), "cudaGraphDestroy after synchronize error");
            return false;
        }
    }
    t1 = clock_type::now();
    std::printf("graph_ms %.6f ms\n", elapsed_ms(t0, t1) / shape_count);
    if (!check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy")) return false;
    if (!check(cudaGraphDestroy(graph), "cudaGraphDestroy")) return false;
    return true;
}

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "auto";
    unsigned int flags = cudaDeviceScheduleAuto;
    if (std::strcmp(mode, "auto") == 0) flags = cudaDeviceScheduleAuto;
    else if (std::strcmp(mode, "spin") == 0) flags = cudaDeviceScheduleSpin;
    else if (std::strcmp(mode, "yield") == 0) flags = cudaDeviceScheduleYield;
    else if (std::strcmp(mode, "blocking") == 0) flags = cudaDeviceScheduleBlockingSync;
    else {
        std::fprintf(stderr, "usage: %s [auto|spin|yield|blocking]\n", argv[0]);
        return 2;
    }
    std::printf("mode %s\n", mode);
    if (!check(cudaSetDeviceFlags(flags), "cudaSetDeviceFlags")) return 1;

    cudaStream_t stream = nullptr;
    unsigned int * data = nullptr;
    if (!check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) return 1;
    if (!check(cudaMalloc(&data, 4096), "cudaMalloc")) {
        check(cudaStreamDestroy(stream), "cudaStreamDestroy after allocation error");
        return 1;
    }
    if (!check(cudaMemsetAsync(data, 0, 4096, stream), "cudaMemsetAsync")) {
        check(cudaFree(data), "cudaFree after memset error");
        check(cudaStreamDestroy(stream), "cudaStreamDestroy after memset error");
        return 1;
    }
    empty_kernel<<<1, 32, 0, stream>>>();
    if (!check(cudaStreamSynchronize(stream), "warmup synchronize")) {
        check(cudaFree(data), "cudaFree after warmup error");
        check(cudaStreamDestroy(stream), "cudaStreamDestroy after warmup error");
        return 1;
    }

    const bool ok = run_measurements(stream, data);
    bool cleanup_ok = true;
    cleanup_ok = check(cudaFree(data), "cudaFree") && cleanup_ok;
    cleanup_ok = check(cudaStreamDestroy(stream), "cudaStreamDestroy") && cleanup_ok;
    return ok && cleanup_ok ? 0 : 1;
}
