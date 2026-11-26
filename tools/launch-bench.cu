#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

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

static void print_stats(const char * name, const char * unit, const std::vector<double> & samples) {
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    const double median = sorted.size() % 2
        ? sorted[sorted.size() / 2]
        : (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0;
    std::printf("%s mean %.6f %s median %.6f %s min %.6f %s\n",
        name, mean, unit, median, unit, sorted.front(), unit);
}

static bool run_measurements(cudaStream_t * stream_ptr, unsigned int * data, unsigned int * expected_increments) {
    cudaStream_t stream = *stream_ptr;
    // Warm both kernels and bring the device out of its idle state before collecting samples.
    for (int i = 0; i < 1024; ++i) empty_kernel<<<1, 32, 0, stream>>>();
    if (!check(cudaGetLastError(), "empty-kernel warmup launch")) return false;
    for (int i = 0; i < 10; ++i) {
        if (!launch_shape(stream, data, 0)) return false;
        if (!check(cudaGetLastError(), "touch-kernel warmup launch")) return false;
        if (!check(cudaStreamSynchronize(stream), "touch-kernel warmup synchronize")) return false;
        *expected_increments += 300;
    }
    if (!check(cudaStreamSynchronize(stream), "empty-kernel warmup synchronize")) return false;

    const int burst_count = 50;
    const int launches_per_burst = 1024;
    std::vector<double> queue_samples;
    std::vector<double> drain_samples;
    for (int burst = 0; burst < burst_count; ++burst) {
        const auto start = clock_type::now();
        for (int i = 0; i < launches_per_burst; ++i) empty_kernel<<<1, 32, 0, stream>>>();
        const auto queued = clock_type::now();
        if (!check(cudaGetLastError(), "queued empty-kernel launches")) return false;
        if (!check(cudaStreamSynchronize(stream), "queue burst synchronize")) return false;
        const auto drained = clock_type::now();
        queue_samples.push_back(elapsed_ms(start, queued) * 1000.0 / launches_per_burst);
        drain_samples.push_back(elapsed_ms(queued, drained) * 1000.0);
    }
    print_stats("launch_queue_us", "us/launch", queue_samples);
    print_stats("drain_us", "us/burst", drain_samples);

    const int roundtrip_count = 2000;
    auto t0 = clock_type::now();
    for (int i = 0; i < roundtrip_count; ++i) {
        empty_kernel<<<1, 32, 0, stream>>>();
        if (!check(cudaStreamSynchronize(stream), "roundtrip synchronize")) return false;
    }
    if (!check(cudaGetLastError(), "roundtrip launches")) return false;
    auto t1 = clock_type::now();
    std::printf("roundtrip_us %.6f us\n", elapsed_ms(t0, t1) * 1000.0 / roundtrip_count);

    const int shape_count = 200;
    std::vector<double> decode_samples;
    for (int i = 0; i < shape_count; ++i) {
        const auto start = clock_type::now();
        if (!launch_shape(stream, data, 0)) return false;
        if (!check(cudaGetLastError(), "decode-shape launches")) return false;
        if (!check(cudaStreamSynchronize(stream), "decode shape synchronize")) return false;
        *expected_increments += 300;
        decode_samples.push_back(elapsed_ms(start, clock_type::now()));
    }
    print_stats("decode_shape_ms", "ms", decode_samples);

    const int flush_counts[] = { 1, 4, 16, 64 };
    for (int n : flush_counts) {
        std::vector<double> flush_samples;
        for (int i = 0; i < shape_count; ++i) {
            const auto start = clock_type::now();
            if (!launch_shape(stream, data, n)) return false;
            if (!check(cudaGetLastError(), "flush-probe launches")) return false;
            if (!check(cudaStreamSynchronize(stream), "flush probe synchronize")) return false;
            *expected_increments += 300;
            flush_samples.push_back(elapsed_ms(start, clock_type::now()));
        }
        char name[32];
        std::snprintf(name, sizeof(name), "flush_%d_ms", n);
        print_stats(name, "ms", flush_samples);
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaError_t status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    bool capture_launches_ok = true;
    if (status == cudaSuccess) {
        for (int i = 0; i < 300; ++i) touch_4k<<<1, 1024, 0, stream>>>(data);
        status = cudaGetLastError();
        if (status != cudaSuccess) capture_launches_ok = false;
        const cudaError_t end_status = cudaStreamEndCapture(stream, &graph);
        if (end_status != cudaSuccess) {
            if (graph) {
                check(cudaGraphDestroy(graph), "cudaGraphDestroy after end-capture error");
                graph = nullptr;
            }
            check(cudaStreamDestroy(stream), "cudaStreamDestroy after end-capture error");
            stream = nullptr;
            if (!check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate after end-capture error")) return false;
            *stream_ptr = stream;
        }
        if (status == cudaSuccess) status = end_status;
    }
    if (status != cudaSuccess || !capture_launches_ok) {
        std::printf("graph_error %s\n", cudaGetErrorString(status));
        if (graph) check(cudaGraphDestroy(graph), "cudaGraphDestroy after graph error");
        return true;
    }

    size_t node_count = 0;
    status = cudaGraphGetNodes(graph, nullptr, &node_count);
    if (status != cudaSuccess || node_count != 300) {
        std::printf("graph_error expected 300 nodes, got %llu (%s)\n",
            static_cast<unsigned long long>(node_count), cudaGetErrorString(status));
        if (graph) check(cudaGraphDestroy(graph), "cudaGraphDestroy after node-count error");
        return true;
    }

    const auto instantiate_start = clock_type::now();
    status = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    const auto instantiate_end = clock_type::now();
    if (status != cudaSuccess) {
        std::printf("graph_error %s\n", cudaGetErrorString(status));
        if (graph_exec) check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy after graph error");
        if (graph) check(cudaGraphDestroy(graph), "cudaGraphDestroy after graph error");
        return true;
    }
    std::printf("graph_instantiate_ms %.6f ms\n", elapsed_ms(instantiate_start, instantiate_end));

    if (!check(cudaGraphLaunch(graph_exec, stream), "graph warmup launch") ||
        !check(cudaGetLastError(), "graph warmup launch status") ||
        !check(cudaStreamSynchronize(stream), "graph warmup synchronize")) {
        check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy after warmup error");
        check(cudaGraphDestroy(graph), "cudaGraphDestroy after warmup error");
        return false;
    }
    *expected_increments += 300;

    std::vector<double> graph_samples;
    for (int i = 0; i < shape_count; ++i) {
        const auto start = clock_type::now();
        if (!check(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch") ||
            !check(cudaGetLastError(), "cudaGraphLaunch status") ||
            !check(cudaStreamSynchronize(stream), "graph synchronize")) {
            check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy after launch error");
            check(cudaGraphDestroy(graph), "cudaGraphDestroy after launch error");
            return false;
        }
        *expected_increments += 300;
        graph_samples.push_back(elapsed_ms(start, clock_type::now()));
    }
    print_stats("graph_ms", "ms", graph_samples);
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
    if (!check(cudaMemsetAsync(data, 0, 4096, stream), "cudaMemsetAsync") ||
        !check(cudaStreamSynchronize(stream), "initial memset synchronize")) {
        check(cudaFree(data), "cudaFree after memset error");
        check(cudaStreamDestroy(stream), "cudaStreamDestroy after memset error");
        return 1;
    }
    unsigned int expected_increments = 0;
    const bool ok = run_measurements(&stream, data, &expected_increments);

    bool data_ok = false;
    unsigned int values[1024] = {};
    if (check(cudaMemcpy(values, data, sizeof(values), cudaMemcpyDeviceToHost), "readback 4 KiB buffer")) {
        data_ok = true;
        for (unsigned int value : values) {
            if (value != expected_increments) {
                data_ok = false;
                break;
            }
        }
    }
    std::printf("check %s (expected increments %u)\n", data_ok ? "ok" : "FAIL", expected_increments);

    bool cleanup_ok = true;
    cleanup_ok = check(cudaFree(data), "cudaFree") && cleanup_ok;
    cleanup_ok = check(cudaStreamDestroy(stream), "cudaStreamDestroy") && cleanup_ok;
    return ok && data_ok && cleanup_ok ? 0 : 1;
}
