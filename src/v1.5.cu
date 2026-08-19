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

// ============================================================
// 取 lane_id 的某一个 bit
// ============================================================
__device__ __forceinline__ unsigned bfe(unsigned lane_id, unsigned pos)
{
    unsigned res;
    asm("bfe.u32 %0,%1,%2,%3;" : "=r"(res) : "r"(lane_id), "r"(pos), "r"(1));
    return res;
}

// ============================================================
// Bitonic Sort 中的一次 xor compare-exchange
// ============================================================
__device__ __forceinline__ int xor_swap(int x, int mask, int dir)
{
    int y = __shfl_xor_sync(FULL_MASK, x, mask, WARP_SIZE);
    return ((x < y) == dir) ? y : x;
}

// ============================================================
// 32-element Warp Bitonic Sort
//
// 输入：
// 每个 lane 一个 A 元素
//
// 输出：
// lane0  : sorted_A[0]
// lane1  : sorted_A[1]
// ...
// lane31 : sorted_A[31]
// ============================================================
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

// ============================================================
// Warp Binary Search
//
// 每个 lane 有自己的 target = B[lane]
//
// sorted_a 分布在整个 Warp：
//
// lane0  -> A[0]
// lane1  -> A[1]
// ...
// lane31 -> A[31]
//
// 每一轮通过 __shfl_sync 读取 mid 位置
//
// A 长度固定 32，所以固定 6 rounds
// ============================================================
__device__ __forceinline__ bool warp_binary_search(int sorted_a, int target)
{
    int left = 0;
    int right = WARP_SIZE;
    bool existed = false;
#pragma unroll
    for (int step = 0; step < 6; ++step) {
        bool active = (!existed && left < right);
        int mid = active ? ((left + right) >> 1) : 0;
        // 整个 Warp 都执行
        int value = __shfl_sync(FULL_MASK, sorted_a, mid);
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

// ============================================================
// 找 mask 中第 k 个置位 bit
//
// k 从 0 开始
//
// 示例：
//
// mask:
// bit位置  0 1 2 3 4 5 ...
//          0 1 0 1 0 1
//
// k=0 -> lane1
// k=1 -> lane3
// k=2 -> lane5
//
// PTX fns 的 offset 从 1 开始，所以 offset=k+1
// ============================================================
__device__ __forceinline__ unsigned find_nth_set_bit(unsigned mask, int k)
{
    unsigned pos;
    unsigned base = 0;
    int offset = k + 1;
    asm volatile("fns.b32 %0, %1, %2, %3;"
                 : "=r"(pos)
                 : "r"(mask), "r"(base), "r"(offset));
    return pos;
}

// ============================================================
// 初始化输入
//
// A:
// 0,2,4,...,62 的 permutation
//
// 每个 B:
// 偶数 lane -> 一定存在于 A
// 奇数 lane -> 一定不存在于 A
//
// 因此每个 B 最终保留 16 个元素
// ============================================================
__global__ void init_input_kernel(int* __restrict__ input_a,
                                  int* __restrict__ input_b, size_t num_groups)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = num_groups * WARP_SIZE;
    if (idx >= total) return;
    size_t group_id = idx / WARP_SIZE;
    int lane = static_cast<int>(idx % WARP_SIZE);
    // --------------------------------------------------------
    // A
    // --------------------------------------------------------
    int a_perm = (lane * 17) & 31;
    input_a[group_id * WARP_SIZE + lane] = a_perm * 2;
    // --------------------------------------------------------
    // B0 ... BN
    // --------------------------------------------------------
#pragma unroll
    for (int b_id = 0; b_id < NUM_B_LISTS; ++b_id) {
        size_t b_idx = (group_id * NUM_B_LISTS + b_id) * WARP_SIZE + lane;
        if ((lane & 1) == 0) {
            // 偶数 lane：
            // 构造一定存在于 A 的值
            int a_lane = (lane * 7 + b_id * 3) & 31;
            int perm = (a_lane * 17) & 31;
            input_b[b_idx] = perm * 2;
        } else {
            // 奇数 lane：
            // 一定不存在于 A
            input_b[b_idx] = 4096 + b_id * 128 + lane;
        }
    }
}

// ============================================================
// V2 Kernel
//
// 1A - N B
//
// A:
// Load once
// -> Bitonic Sort once
// -> sorted A stays in registers
//
// 每个 B:
//
// Binary Search
// -> valid_mask
// -> destination lane 找 source lane
// -> Shuffle Gather
// -> dense B in registers
// -> 连续写 Shared
//
// 和 V1 的区别只在 Compact 方法。
// ============================================================
__global__ __launch_bounds__(BLOCK_SIZE) void v2_register_gather_compact_kernel(
    const int* __restrict__ input_a, const int* __restrict__ input_b,
    int* __restrict__ output_count, int* __restrict__ output_checksum)
{
    // --------------------------------------------------------
    // 每个 Warp 32 个 int
    //
    // 16 warps/block × 32 × 4B
    // = 2048 bytes
    // --------------------------------------------------------
    __shared__ int s_compact[BLOCK_SIZE];
    int tx = threadIdx.x;
    int warp_id = tx >> 5;
    int lane = tx & 31;
    size_t group_id = static_cast<size_t>(blockIdx.x) * NUM_WARPS + warp_id;
    int shared_base = warp_id * WARP_SIZE;
    // ========================================================
    // Step 1
    // Load A[32]
    //
    // 每个 lane 加载一个 A
    // ========================================================
    int a_value = input_a[group_id * WARP_SIZE + lane];
    // ========================================================
    // Step 2
    // Sort A once
    //
    // sorted A 分布在 Warp registers
    // ========================================================
    int sorted_a = warp_bitonic_sort(a_value, lane);
    // ========================================================
    // B0 ... BN
    // ========================================================
#pragma unroll
    for (int b_id = 0; b_id < NUM_B_LISTS; ++b_id) {
        // ====================================================
        // Step 3
        // Load B
        // ====================================================
        size_t b_idx = (group_id * NUM_B_LISTS + b_id) * WARP_SIZE + lane;
        int b = input_b[b_idx];
        // ====================================================
        // Step 4
        // Binary Search
        // ====================================================
        bool duplicated = warp_binary_search(sorted_a, b);
        // ====================================================
        // Step 5
        // Ballot
        //
        // bit i = 1
        // 表示 B[i] 不在 A 中
        // ====================================================
        unsigned valid_mask = __ballot_sync(FULL_MASK, !duplicated);
        // ====================================================
        // Step 6
        // unique count
        // ====================================================
        int count = __popc(valid_mask);
        // ====================================================
        // Step 7
        // V2 Compact
        //
        // 注意：
        //
        // V1 是：
        //
        // source lane
        // -> rank
        // -> scatter
        //
        //
        // V2 是：
        //
        // destination lane
        // -> 找第 k 个 valid source
        // -> gather
        //
        //
        // lane0 负责 compact[0]
        // lane1 负责 compact[1]
        // ...
        // ====================================================
        // 默认设为 0。
        //
        // lane >= count 时这个值无意义，
        // 但后面的 SHFL 仍然必须由整个 Warp 执行。
        unsigned src_lane = 0;
        // ----------------------------------------------------
        // 只有真正的 destination lane
        // 需要寻找 source lane
        // ----------------------------------------------------
        if (lane < count) {
            src_lane = find_nth_set_bit(valid_mask, lane);
        }
        // ====================================================
        // 最关键的修复：
        //
        // 这里不能写：
        //
        // if (lane < count)
        //     __shfl_sync(FULL_MASK,...)
        //
        // 必须让整个 Warp 都执行 SHFL。
        //
        //
        // lane >= count：
        // src_lane=0
        // 会拿 B[0]
        //
        // 但这个结果最终不会使用，所以无所谓。
        // ====================================================
        int dense_b = __shfl_sync(FULL_MASK, b, src_lane, WARP_SIZE);
        // ====================================================
        // 现在：
        //
        // lane0         dense_B[0]
        // lane1         dense_B[1]
        // ...
        // lane(count-1) dense_B[count-1]
        //
        // dense B 已经在 Registers 中形成
        // ====================================================
        // ====================================================
        // Step 8
        // 连续写 Shared
        //
        // 注意这里不是 scatter。
        //
        // lane0 -> shared[0]
        // lane1 -> shared[1]
        // ...
        // ====================================================
        if (lane < count) {
            s_compact[shared_base + lane] = dense_b;
        }
        // Shared write -> Shared read
        // 必须保证 warp 内顺序
        __syncwarp(FULL_MASK);
        // ====================================================
        // 下面这一部分和 V1 benchmark 保持一致。
        //
        // 它不是 remove_duplicates 的核心，
        // 只是为了真的消费 compact 结果，
        // 防止编译器把前面的工作优化掉，
        // 同时做 correctness check。
        // ====================================================
        int compact_value = lane < count ? s_compact[shared_base + lane] : 0;
        // ----------------------------------------------------
        // Warp checksum reduction
        // ----------------------------------------------------
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            compact_value +=
                __shfl_down_sync(FULL_MASK, compact_value, offset, WARP_SIZE);
        }
        // ----------------------------------------------------
        // lane0 写输出
        // ----------------------------------------------------
        if (lane == 0) {
            size_t out_idx = group_id * NUM_B_LISTS + b_id;
            output_count[out_idx] = count;
            output_checksum[out_idx] = compact_value;
        }
        // 防止下一个 B 覆盖 shared，
        // 当前 warp 还没读完。
        __syncwarp(FULL_MASK);
    }
}

// ============================================================
// main
// ============================================================
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
    // ========================================================
    // Init
    // ========================================================
    constexpr int INIT_THREADS = 256;
    size_t init_total = num_groups * WARP_SIZE;
    int init_blocks =
        static_cast<int>((init_total + INIT_THREADS - 1) / INIT_THREADS);
    init_input_kernel<<<init_blocks, INIT_THREADS>>>(d_a, d_b, num_groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // ========================================================
    // Warmup
    // ========================================================
    constexpr int WARMUP = 10;
    for (int i = 0; i < WARMUP; ++i) {
        v2_register_gather_compact_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            d_a, d_b, d_count, d_checksum);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // ========================================================
    // Benchmark
    // ========================================================
    constexpr int REPEAT = 50;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPEAT; ++i) {
        v2_register_gather_compact_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            d_a, d_b, d_count, d_checksum);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / REPEAT;
    // ========================================================
    // Copy results
    // ========================================================
    std::vector<int> h_count(num_outputs);
    std::vector<int> h_checksum(num_outputs);
    CUDA_CHECK(cudaMemcpy(h_count.data(), d_count, output_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_checksum.data(), d_checksum, output_bytes,
                          cudaMemcpyDeviceToHost));
    // ========================================================
    // Correctness
    // ========================================================
    size_t errors = 0;
    for (size_t group = 0; group < num_groups; ++group) {
        for (int b_id = 0; b_id < NUM_B_LISTS; ++b_id) {
            size_t idx = group * NUM_B_LISTS + b_id;
            // 每个 B：
            //
            // 奇数 lane 1,3,...31 保留
            //
            // 16 × (4096 + b_id*128)
            //
            // odd lanes sum:
            //
            // 1+3+...+31 = 256
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
    // ========================================================
    // Logical traffic
    //
    // 为了和 V1-real 完全一样：
    //
    // A:
    // 128 B / group
    //
    // 每个 B:
    //
    // input B       128 B
    // output count    4 B
    // checksum        4 B
    //
    // 136 B / B
    // ========================================================
    double bytes_per_group = 128.0 + NUM_B_LISTS * 136.0;
    double total_bytes = bytes_per_group * num_groups;
    double seconds = avg_ms / 1000.0;
    double effective_bw = total_bytes / seconds / 1e9;
    double groups_per_sec = num_groups / seconds;
    double pairs_per_sec = num_groups * NUM_B_LISTS / seconds;
    printf("\n");
    printf("V2 1A-NB Register Gather Compact\n");
    printf("Average time    : %.6f ms\n", avg_ms);
    printf("Groups/s        : %.3f M\n", groups_per_sec / 1e6);
    printf("A-B pairs/s     : %.3f M\n", pairs_per_sec / 1e6);
    printf("Logical traffic : %.3f GB\n", total_bytes / 1e9);
    printf("Effective BW    : %.3f GB/s\n", effective_bw);
    printf("Search stages   : %d / group\n", 15 + NUM_B_LISTS * 6);
    printf("Wrong results   : %zu\n", errors);
    // ========================================================
    // Cleanup
    // ========================================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_checksum));
    return 0;
}
