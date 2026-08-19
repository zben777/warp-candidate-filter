#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef NUM_B_LISTS
#define NUM_B_LISTS 4
#endif

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 512;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

constexpr int NUM_BLOCKS = 256 * 1024;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__device__ __forceinline__ unsigned lane_mask_lt(int lane)
{
    return lane == 0 ? 0u : ((1u << lane) - 1u);
}

__global__ void init_input_kernel(int* __restrict__ input_a,
                                  int* __restrict__ input_b, size_t num_groups)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = num_groups * WARP_SIZE;
    if (idx >= total) {
        return;
    }
    size_t group_id = idx / WARP_SIZE;
    int lane = static_cast<int>(idx % WARP_SIZE);
    // A[32]
    // 是 0,2,4,...,62 的一个 permutation
    int a_perm = (lane * 17) & 31;
    input_a[group_id * WARP_SIZE + lane] = a_perm * 2;
    // 每个 B：
    //
    // 偶数 lane：
    // 一定存在于 A
    //
    // 奇数 lane：
    // 一定不存在于 A
    //
    // 所以每个 B 最终保留 16 个元素
#pragma unroll
    for (int b_id = 0; b_id < NUM_B_LISTS; ++b_id) {
        size_t b_idx = (group_id * NUM_B_LISTS + b_id) * WARP_SIZE + lane;
        if ((lane & 1) == 0) {
            int a_lane = (lane * 7 + b_id * 3) & 31;
            int perm = (a_lane * 17) & 31;
            input_b[b_idx] = perm * 2;
        } else {
            input_b[b_idx] = 4096 + b_id * 128 + lane;
        }
    }
}

__global__ __launch_bounds__(BLOCK_SIZE) void brute_force_shared_kernel(
    const int* __restrict__ input_a, const int* __restrict__ input_b,
    int* __restrict__ output_count, int* __restrict__ output_checksum)
{
    // 每个 Warp 一份 A[32]
    //
    // 16 warps/block × 32 int
    // = 512 int
    // = 2048 B
    __shared__ int s_a[BLOCK_SIZE];
    // 每个 Warp 一份 compact buffer
    //
    // 同样 2048 B
    __shared__ int s_compact[BLOCK_SIZE];
    int tx = threadIdx.x;
    int warp_id = tx >> 5;
    int lane = tx & 31;
    size_t group_id = static_cast<size_t>(blockIdx.x) * NUM_WARPS + warp_id;
    int shared_base = warp_id * WARP_SIZE;
    // 每个 lane 从 Global Memory
    // 加载一个 A 元素
    //
    // 然后写入当前 Warp 的 Shared Memory
    int a_value = input_a[group_id * WARP_SIZE + lane];
    s_a[shared_base + lane] = a_value;
    // 保证当前 Warp 的 A[32]
    // 已经全部写入 Shared
    __syncwarp();
    // 使用 volatile，
    // 避免编译器把 Shared A 过度缓存成
    // register 常量路径。
    volatile int* warp_a = s_a + shared_base;
#pragma unroll
    for (int b_id = 0; b_id < NUM_B_LISTS; ++b_id) {
        size_t b_idx = (group_id * NUM_B_LISTS + b_id) * WARP_SIZE + lane;
        int b = input_b[b_idx];
        bool duplicated = false;
        // 与 Register + Shuffle 版本的核心区别就在这里：
        //
        // V0-register:
        // candidate = __shfl_sync(..., a_value, src)
        //
        // V0-shared:
        // candidate = warp_a[src]
        //
        // 一个 Warp 的 32 个 lane 在同一轮
        // 都读取相同 Shared 地址，
        // 属于 Shared Memory broadcast。
#pragma unroll
        for (int src = 0; src < WARP_SIZE; ++src) {
            int candidate = warp_a[src];
            duplicated |= (b == candidate);
        }
        unsigned valid_mask = __ballot_sync(0xffffffffu, !duplicated);
        int count = __popc(valid_mask);
        int rank = __popc(valid_mask & lane_mask_lt(lane));
        // 与 V0-real 完全相同：
        // unique B compact 到 Shared Memory
        if (!duplicated) {
            s_compact[shared_base + rank] = b;
        }
        __syncwarp();
        // 与 V0-real 完全相同：
        // 模拟 compact 后的数据继续
        // 在 kernel 内被消费
        int compact_value = lane < count ? s_compact[shared_base + lane] : 0;
        // Warp reduction 生成 checksum，
        // 防止 compiler 删除 compact 路径
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            compact_value +=
                __shfl_down_sync(0xffffffffu, compact_value, offset);
        }
        if (lane == 0) {
            size_t out_idx = group_id * NUM_B_LISTS + b_id;
            output_count[out_idx] = count;
            output_checksum[out_idx] = compact_value;
        }
        // 防止下一个 B 覆盖 s_compact
        // 时还有 lane 没消费完
        __syncwarp();
    }
}

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const size_t num_groups = static_cast<size_t>(NUM_BLOCKS) * NUM_WARPS;
    const size_t num_a_elements = num_groups * WARP_SIZE;
    const size_t num_b_elements = num_groups * NUM_B_LISTS * WARP_SIZE;
    const size_t num_outputs = num_groups * NUM_B_LISTS;
    const size_t a_bytes = num_a_elements * sizeof(int);
    const size_t b_bytes = num_b_elements * sizeof(int);
    const size_t output_bytes = num_outputs * sizeof(int);
    printf("GPU             : %s\n", prop.name);
    printf("Grid            : <<<%d, %d>>>\n", NUM_BLOCKS, BLOCK_SIZE);
    printf("Warps/block     : %d\n", NUM_WARPS);
    printf("Groups          : %zu\n", num_groups);
    printf("NUM_B_LISTS     : %d\n", NUM_B_LISTS);
    printf("A memory        : %.2f MiB\n", a_bytes / 1024.0 / 1024.0);
    printf("B memory        : %.2f MiB\n", b_bytes / 1024.0 / 1024.0);
    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_count = nullptr;
    int* d_checksum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_count, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_checksum, output_bytes));
    constexpr int INIT_THREADS = 256;
    size_t init_total = num_groups * WARP_SIZE;
    int init_blocks =
        static_cast<int>((init_total + INIT_THREADS - 1) / INIT_THREADS);
    init_input_kernel<<<init_blocks, INIT_THREADS>>>(d_a, d_b, num_groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    constexpr int WARMUP = 10;
    for (int i = 0; i < WARMUP; ++i) {
        brute_force_shared_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_count,
                                                              d_checksum);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    constexpr int REPEAT = 50;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPEAT; ++i) {
        brute_force_shared_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_count,
                                                              d_checksum);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / REPEAT;
    std::vector<int> h_count(num_outputs);
    std::vector<int> h_checksum(num_outputs);
    CUDA_CHECK(cudaMemcpy(h_count.data(), d_count, output_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_checksum.data(), d_checksum, output_bytes,
                          cudaMemcpyDeviceToHost));
    size_t errors = 0;
    for (size_t group = 0; group < num_groups; ++group) {
        for (int b_id = 0; b_id < NUM_B_LISTS; ++b_id) {
            size_t idx = group * NUM_B_LISTS + b_id;
            int expected_checksum = 16 * 4096 + 16 * b_id * 128 + 256;
            if (h_count[idx] != 16 || h_checksum[idx] != expected_checksum) {
                if (errors < 10) {
                    printf("Mismatch group=%zu b=%d count=%d checksum=%d "
                           "expected_checksum=%d\n",
                           group, b_id, h_count[idx], h_checksum[idx],
                           expected_checksum);
                }
                ++errors;
            }
        }
    }
    // Global Memory traffic 与 V0-real 完全相同
    //
    // A input:
    // 128 B
    //
    // 每个 B:
    // input       128 B
    // count         4 B
    // checksum      4 B
    //
    // = 136 B / B
    //
    // Shared Memory traffic 不计入这里的
    // Logical Global Traffic。
    double bytes_per_group = 128.0 + NUM_B_LISTS * 136.0;
    double total_bytes = bytes_per_group * num_groups;
    double seconds = avg_ms / 1000.0;
    double effective_bw = total_bytes / seconds / 1e9;
    double groups_per_sec = num_groups / seconds;
    double pairs_per_sec = num_groups * NUM_B_LISTS / seconds;
    printf("\n");
    printf("V0-Shared 1A-NB Shared Broadcast Brute Force\n");
    printf("Average time    : %.6f ms\n", avg_ms);
    printf("Groups/s        : %.3f M\n", groups_per_sec / 1e6);
    printf("A-B pairs/s     : %.3f M\n", pairs_per_sec / 1e6);
    printf("Logical traffic : %.3f GB\n", total_bytes / 1e9);
    printf("Effective BW    : %.3f GB/s\n", effective_bw);
    printf("Search rounds   : %d / group\n", NUM_B_LISTS * 32);
    printf("Wrong results   : %zu\n", errors);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_checksum));
    return 0;
}
