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

constexpr unsigned FULL_MASK = 0xffffffffu;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// 取 lane_id 的某一个 bit。
// 用于保持和 V1 完全相同的 Bitonic Sort。
__device__ __forceinline__ unsigned bfe(unsigned lane_id, unsigned pos)
{
    unsigned res;
    asm("bfe.u32 %0,%1,%2,%3;" : "=r"(res) : "r"(lane_id), "r"(pos), "r"(1));
    return res;
}

// Bitonic Sort 中的一次 compare-exchange。
__device__ __forceinline__ int xor_swap(int x, int mask, int dir)
{
    int y = __shfl_xor_sync(FULL_MASK, x, mask, WARP_SIZE);
    return ((x < y) == dir) ? y : x;
}

// 32 个 lane 一起完成 A[32] 的 Bitonic Sort。
// 排序结束后：
// lane0  -> sorted_A[0]
// lane1  -> sorted_A[1]
// ...
// lane31 -> sorted_A[31]
__device__ __forceinline__ int warp_bitonic_sort(int element, int lane)
{
    element = xor_swap(element, 0x01, bfe(lane, 1) ^ bfe(lane, 0));
    element = xor_swap(element, 0x02, bfe(lane, 2) ^ bfe(lane, 1));
    element = xor_swap(element, 0x01, bfe(lane, 2) ^ bfe(lane, 0));
    element = xor_swap(element, 0x04, bfe(lane, 3) ^ bfe(lane, 2));
    element = xor_swap(element, 0x02, bfe(lane, 3) ^ bfe(lane, 1));
    element = xor_swap(element, 0x01, bfe(lane, 3) ^ bfe(lane, 0));
    element = xor_swap(element, 0x08, bfe(lane, 4) ^ bfe(lane, 3));
    element = xor_swap(element, 0x04, bfe(lane, 4) ^ bfe(lane, 2));
    element = xor_swap(element, 0x02, bfe(lane, 4) ^ bfe(lane, 1));
    element = xor_swap(element, 0x01, bfe(lane, 4) ^ bfe(lane, 0));
    element = xor_swap(element, 0x10, bfe(lane, 4));
    element = xor_swap(element, 0x08, bfe(lane, 3));
    element = xor_swap(element, 0x04, bfe(lane, 2));
    element = xor_swap(element, 0x02, bfe(lane, 1));
    element = xor_swap(element, 0x01, bfe(lane, 0));
    return element;
}

// V2 核心：2-way Binary Search ILP
//
// target0 和 target1 是两个独立 B list 中，
// 当前 lane 持有的两个 B 元素。
//
// V1 是：
// B0 step0 -> step1 -> ... -> step5
// B1 step0 -> step1 -> ... -> step5
//
// V2 是：
// B0 step0
// B1 step0
// B0 step1
// B1 step1
// ...
//
// 更重要的是，这里每一轮会先发出两个独立的
// __shfl_sync，然后再分别更新两套搜索状态。
__device__ __forceinline__ void
warp_binary_search_2way(int sorted_a, int target0, int target1, bool& existed0,
                        bool& existed1)
{
    int left0 = 0;
    int right0 = WARP_SIZE;
    int left1 = 0;
    int right1 = WARP_SIZE;
    existed0 = false;
    existed1 = false;
#pragma unroll
    for (int step = 0; step < 6; ++step) {
        bool active0 = (!existed0 && left0 < right0);
        bool active1 = (!existed1 && left1 < right1);
        int mid0 = active0 ? ((left0 + right0) >> 1) : 0;
        int mid1 = active1 ? ((left1 + right1) >> 1) : 0;
        // 两个独立的 warp shuffle。
        // 这里是 V2 想制造 ILP 的关键位置。
        int value0 = __shfl_sync(FULL_MASK, sorted_a, mid0, WARP_SIZE);
        int value1 = __shfl_sync(FULL_MASK, sorted_a, mid1, WARP_SIZE);
        // 更新 B0 的 Binary Search 状态。
        if (active0) {
            if (value0 == target0) {
                existed0 = true;
            } else if (target0 > value0) {
                left0 = mid0 + 1;
            } else {
                right0 = mid0;
            }
        }
        // 更新 B1 的 Binary Search 状态。
        if (active1) {
            if (value1 == target1) {
                existed1 = true;
            } else if (target1 > value1) {
                left1 = mid1 + 1;
            } else {
                right1 = mid1;
            }
        }
    }
}

// 如果 NUM_B_LISTS 是奇数，最后一个 B 使用这个。
// 对 4B benchmark 不会走这里。
__device__ __forceinline__ bool warp_binary_search_1way(int sorted_a,
                                                        int target)
{
    int left = 0;
    int right = WARP_SIZE;
    bool existed = false;
#pragma unroll
    for (int step = 0; step < 6; ++step) {
        bool active = (!existed && left < right);
        int mid = active ? ((left + right) >> 1) : 0;
        int value = __shfl_sync(FULL_MASK, sorted_a, mid, WARP_SIZE);
        if (active) {
            if (value == target) {
                existed = true;
            } else if (target > value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
    }
    return existed;
}

// 保留 V1 原来的 Compact 方法。
// 这部分 V2 不优化。
//
// valid
// -> ballot
// -> popc rank
// -> Shared Scatter
// -> dense B
// -> benchmark checksum
__device__ __forceinline__ void compact_and_consume(int b, bool duplicated,
                                                    int lane, int shared_base,
                                                    int* s_compact, int& count,
                                                    int& checksum)
{
    unsigned valid_mask = __ballot_sync(FULL_MASK, !duplicated);
    count = __popc(valid_mask);
    // lane_mask_lt:
    //
    // lane0  -> 000...000
    // lane1  -> 000...001
    // lane2  -> 000...011
    // ...
    //
    // 用来计算当前 valid lane 前面
    // 有多少个 valid lane。
    unsigned lane_mask_lt = (1u << lane) - 1u;
    int rank = __popc(valid_mask & lane_mask_lt);
    // 和 V1 完全相同：
    // 每个 valid source lane
    // scatter 到自己的 dense rank。
    if (!duplicated) {
        s_compact[shared_base + rank] = b;
    }
    __syncwarp(FULL_MASK);
    // 下面是 benchmark 消费 compact 结果。
    // 不属于 remove_duplicates 核心算法。
    int compact_value = lane < count ? s_compact[shared_base + lane] : 0;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        compact_value +=
            __shfl_down_sync(FULL_MASK, compact_value, offset, WARP_SIZE);
    }
    checksum = compact_value;
    // 保证当前 B 的 Shared 数据已经消费完成，
    // 才能让下一个 B 覆盖同一个区域。
    __syncwarp(FULL_MASK);
}

// 初始化测试输入。
//
// A 是 0,2,4,...,62 的 permutation。
//
// 每个 B：
// 偶数 lane -> 一定与 A 重复
// 奇数 lane -> 一定不在 A 中
//
// 所以每个 B 最后应该留下 16 个元素。
__global__ void init_input_kernel(int* __restrict__ input_a,
                                  int* __restrict__ input_b, size_t num_groups)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = num_groups * WARP_SIZE;
    if (idx >= total) return;
    size_t group_id = idx / WARP_SIZE;
    int lane = static_cast<int>(idx % WARP_SIZE);
    // A[32]
    int a_perm = (lane * 17) & 31;
    input_a[group_id * WARP_SIZE + lane] = a_perm * 2;
#pragma unroll
    for (int b_id = 0; b_id < NUM_B_LISTS; ++b_id) {
        size_t b_idx = (group_id * NUM_B_LISTS + b_id) * WARP_SIZE + lane;
        if ((lane & 1) == 0) {
            // 偶数 lane 构造一个一定存在于 A 的值。
            int a_lane = (lane * 7 + b_id * 3) & 31;
            int perm = (a_lane * 17) & 31;
            input_b[b_idx] = perm * 2;
        } else {
            // 奇数 lane 一定不在 A 中。
            input_b[b_idx] = 4096 + b_id * 128 + lane;
        }
    }
}

// 正式 V2:
//
// Sort A Once
// +
// 2-way B Binary Search ILP
// +
// 保持 V1 原来的 Shared Scatter Compact
__global__ __launch_bounds__(BLOCK_SIZE) void v2_2way_ilp_kernel(
    const int* __restrict__ input_a, const int* __restrict__ input_b,
    int* __restrict__ output_count, int* __restrict__ output_checksum)
{
    // 和 V1 相同：
    // 每个 Warp 32 个 int。
    //
    // 16 warps × 32 × 4B
    // = 2048 B / block
    __shared__ int s_compact[BLOCK_SIZE];
    int tx = threadIdx.x;
    int warp_id = tx >> 5;
    int lane = tx & 31;
    size_t group_id = static_cast<size_t>(blockIdx.x) * NUM_WARPS + warp_id;
    int shared_base = warp_id * WARP_SIZE;
    // Step 1:
    // A 只从 Global Load 一次。
    int a_value = input_a[group_id * WARP_SIZE + lane];
    // Step 2:
    // A 只 Sort 一次。
    int sorted_a = warp_bitonic_sort(a_value, lane);
    // 每次同时处理两个 B。
#pragma unroll
    for (int b_base = 0; b_base + 1 < NUM_B_LISTS; b_base += 2) {
        int b_id0 = b_base;
        int b_id1 = b_base + 1;
        size_t b_idx0 = (group_id * NUM_B_LISTS + b_id0) * WARP_SIZE + lane;
        size_t b_idx1 = (group_id * NUM_B_LISTS + b_id1) * WARP_SIZE + lane;
        // V2:
        // 两个独立 B 一起加载进 Register。
        int b0 = input_b[b_idx0];
        int b1 = input_b[b_idx1];
        bool duplicated0;
        bool duplicated1;
        // V2 核心：
        // 两条 Binary Search dependency chain
        // 在同一个循环中交错执行。
        warp_binary_search_2way(sorted_a, b0, b1, duplicated0, duplicated1);
        // Compact 部分完全沿用 V1。
        int count0;
        int checksum0;
        compact_and_consume(b0, duplicated0, lane, shared_base, s_compact,
                            count0, checksum0);
        if (lane == 0) {
            size_t out_idx0 = group_id * NUM_B_LISTS + b_id0;
            output_count[out_idx0] = count0;
            output_checksum[out_idx0] = checksum0;
        }
        int count1;
        int checksum1;
        compact_and_consume(b1, duplicated1, lane, shared_base, s_compact,
                            count1, checksum1);
        if (lane == 0) {
            size_t out_idx1 = group_id * NUM_B_LISTS + b_id1;
            output_count[out_idx1] = count1;
            output_checksum[out_idx1] = checksum1;
        }
    }
    // 如果 NUM_B_LISTS 是奇数，
    // 最后剩下一个 B 单独处理。
#if (NUM_B_LISTS % 2) == 1
    constexpr int b_id = NUM_B_LISTS - 1;
    size_t b_idx = (group_id * NUM_B_LISTS + b_id) * WARP_SIZE + lane;
    int b = input_b[b_idx];
    bool duplicated = warp_binary_search_1way(sorted_a, b);
    int count;
    int checksum;
    compact_and_consume(b, duplicated, lane, shared_base, s_compact, count,
                        checksum);
    if (lane == 0) {
        size_t out_idx = group_id * NUM_B_LISTS + b_id;
        output_count[out_idx] = count;
        output_checksum[out_idx] = checksum;
    }
#endif
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
    // 初始化输入。
    constexpr int INIT_THREADS = 256;
    size_t init_total = num_groups * WARP_SIZE;
    int init_blocks =
        static_cast<int>((init_total + INIT_THREADS - 1) / INIT_THREADS);
    init_input_kernel<<<init_blocks, INIT_THREADS>>>(d_a, d_b, num_groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // Warmup。
    constexpr int WARMUP = 10;
    for (int i = 0; i < WARMUP; ++i) {
        v2_2way_ilp_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_count,
                                                       d_checksum);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // Benchmark。
    constexpr int REPEAT = 50;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPEAT; ++i) {
        v2_2way_ilp_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_count,
                                                       d_checksum);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / REPEAT;
    // 拷回 correctness 输出。
    std::vector<int> h_count(num_outputs);
    std::vector<int> h_checksum(num_outputs);
    CUDA_CHECK(cudaMemcpy(h_count.data(), d_count, output_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_checksum.data(), d_checksum, output_bytes,
                          cudaMemcpyDeviceToHost));
    // Correctness check。
    size_t errors = 0;
    for (size_t group = 0; group < num_groups; ++group) {
        for (int b_id = 0; b_id < NUM_B_LISTS; ++b_id) {
            size_t idx = group * NUM_B_LISTS + b_id;
            // 保留下来的都是奇数 lane：
            //
            // 1 + 3 + ... + 31 = 256
            //
            // 每个 B 共 16 个 unique 元素。
            int expected_checksum = 16 * 4096 + 16 * b_id * 128 + 256;
            if (h_count[idx] != 16 || h_checksum[idx] != expected_checksum) {
                if (errors < 10) {
                    printf("Mismatch group=%zu b=%d "
                           "count=%d checksum=%d "
                           "expected_count=16 "
                           "expected_checksum=%d\n",
                           group, b_id, h_count[idx], h_checksum[idx],
                           expected_checksum);
                }
                ++errors;
            }
        }
    }
    // 和 V1 使用完全相同的 Logical Global Traffic。
    //
    // A:
    // 32 int = 128 B
    //
    // 每个 B:
    // B input     = 128 B
    // count       =   4 B
    // checksum    =   4 B
    //
    // 每个 B 总计 136 B。
    double bytes_per_group = 128.0 + NUM_B_LISTS * 136.0;
    double total_bytes = bytes_per_group * num_groups;
    double seconds = avg_ms / 1000.0;
    double effective_bw = total_bytes / seconds / 1e9;
    double groups_per_sec = num_groups / seconds;
    double pairs_per_sec = num_groups * NUM_B_LISTS / seconds;
    printf("\n");
    printf("V2 1A-NB Sort Once + 2-Way B ILP\n");
    printf("Average time    : %.6f ms\n", avg_ms);
    printf("Groups/s        : %.3f M\n", groups_per_sec / 1e6);
    printf("A-B pairs/s     : %.3f M\n", pairs_per_sec / 1e6);
    printf("Logical traffic : %.3f GB\n", total_bytes / 1e9);
    printf("Effective BW    : %.3f GB/s\n", effective_bw);
    printf("Search stages   : %d / group\n", 15 + NUM_B_LISTS * 6);
    printf("B ILP width     : 2\n");
    printf("Wrong results   : %zu\n", errors);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_checksum));
    return 0;
}
