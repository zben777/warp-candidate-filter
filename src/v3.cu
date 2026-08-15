#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef NUM_B_LISTS
#define NUM_B_LISTS 4
#endif

constexpr int WARP_SIZE  = 32;
constexpr int BLOCK_SIZE = 512;
constexpr int NUM_WARPS  = BLOCK_SIZE / WARP_SIZE;
constexpr int NUM_BLOCKS = 256 * 1024;

constexpr unsigned FULL_MASK = 0xffffffffu;

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr,                                                    \
                "CUDA error %s:%d: %s\n",                                  \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)


// 取 lane_id 的某一位。
// 保持和 V1 相同的 Bitonic Sort 实现。
__device__ __forceinline__
unsigned bfe(
    unsigned lane_id,
    unsigned pos)
{
    unsigned res;

    asm(
        "bfe.u32 %0,%1,%2,%3;"
        : "=r"(res)
        : "r"(lane_id),
          "r"(pos),
          "r"(1));

    return res;
}


// Bitonic Sort 的一次 compare-exchange。
__device__ __forceinline__
int xor_swap(
    int x,
    int mask,
    int dir)
{
    int y =
        __shfl_xor_sync(
            FULL_MASK,
            x,
            mask,
            WARP_SIZE);

    return ((x < y) == dir)
               ? y
               : x;
}


// Warp Bitonic Sort。
//
// 输入：
// lane0  -> A[0]
// lane1  -> A[1]
// ...
//
// 输出：
// lane0  -> sorted_A[0]
// lane1  -> sorted_A[1]
// ...
// lane31 -> sorted_A[31]
__device__ __forceinline__
int warp_bitonic_sort(
    int element,
    int lane)
{
    element = xor_swap(
        element,
        0x01,
        bfe(lane, 1) ^ bfe(lane, 0));

    element = xor_swap(
        element,
        0x02,
        bfe(lane, 2) ^ bfe(lane, 1));

    element = xor_swap(
        element,
        0x01,
        bfe(lane, 2) ^ bfe(lane, 0));

    element = xor_swap(
        element,
        0x04,
        bfe(lane, 3) ^ bfe(lane, 2));

    element = xor_swap(
        element,
        0x02,
        bfe(lane, 3) ^ bfe(lane, 1));

    element = xor_swap(
        element,
        0x01,
        bfe(lane, 3) ^ bfe(lane, 0));

    element = xor_swap(
        element,
        0x08,
        bfe(lane, 4) ^ bfe(lane, 3));

    element = xor_swap(
        element,
        0x04,
        bfe(lane, 4) ^ bfe(lane, 2));

    element = xor_swap(
        element,
        0x02,
        bfe(lane, 4) ^ bfe(lane, 1));

    element = xor_swap(
        element,
        0x01,
        bfe(lane, 4) ^ bfe(lane, 0));

    element = xor_swap(
        element,
        0x10,
        bfe(lane, 4));

    element = xor_swap(
        element,
        0x08,
        bfe(lane, 3));

    element = xor_swap(
        element,
        0x04,
        bfe(lane, 2));

    element = xor_swap(
        element,
        0x02,
        bfe(lane, 1));

    element = xor_swap(
        element,
        0x01,
        bfe(lane, 0));

    return element;
}


// V3 核心：Fixed-32 Specialized Search
//
// 目标：
// 不再维护：
//     left
//     right
//     active
//     mid
//     existed
//
// 搜索过程中只维护：
//     pos
//
// pos 表示：
// 当前已经确认的
// “最后一个 < target 的位置”。
//
// 初始：
//     pos = -1
//
// 固定 probe：
//     +16
//     +8
//     +4
//     +2
//     +1
//
// 最后：
//     candidate = pos + 1
//
// 检查 sorted_A[candidate] == target。
//
// 这里总共仍然是 6 次 SHFL：
// 5 次定位 + 1 次最终 equality check。
//
// 优化目标不是大幅减少 SHFL 数量，
// 而是减少通用 Binary Search 的控制和状态维护。
__device__ __forceinline__
bool warp_fixed32_membership(
    int sorted_a,
    int target)
{
    int pos = -1;


    // Step +16
    //
    // pos 最大只能更新到 15。
    {
        int probe =
            pos + 16;

        int value =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe,
                WARP_SIZE);

        pos =
            (value < target)
                ? probe
                : pos;
    }


    // Step +8
    {
        int probe =
            pos + 8;

        int value =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe,
                WARP_SIZE);

        pos =
            (value < target)
                ? probe
                : pos;
    }


    // Step +4
    {
        int probe =
            pos + 4;

        int value =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe,
                WARP_SIZE);

        pos =
            (value < target)
                ? probe
                : pos;
    }


    // Step +2
    {
        int probe =
            pos + 2;

        int value =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe,
                WARP_SIZE);

        pos =
            (value < target)
                ? probe
                : pos;
    }


    // Step +1
    {
        int probe =
            pos + 1;

        int value =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe,
                WARP_SIZE);

        pos =
            (value < target)
                ? probe
                : pos;
    }


    // 经过上面 5 步之后：
    //
    // pos ∈ [-1, 30]
    //
    // 所以：
    //
    // candidate = pos + 1
    //
    // 一定在：
    //
    // [0, 31]
    //
    // 不需要额外 bounds check。
    int candidate =
        pos + 1;


    int value =
        __shfl_sync(
            FULL_MASK,
            sorted_a,
            candidate,
            WARP_SIZE);


    return value == target;
}


// V1 原来的 Shared Compact。
// V3 完全不改这一部分。
//
// duplicated
// ↓
// ballot
// ↓
// valid_mask
// ↓
// popc rank
// ↓
// Shared scatter
// ↓
// dense B
//
// __popc 对 32-bit 输入统计置位 bit 数。:contentReference[oaicite:1]{index=1}
__device__ __forceinline__
void compact_and_consume(
    int b,
    bool duplicated,
    int lane,
    int shared_base,
    int* s_compact,
    int& count,
    int& checksum)
{
    unsigned valid_mask =
        __ballot_sync(
            FULL_MASK,
            !duplicated);


    count =
        __popc(
            valid_mask);


    // 当前 lane 前面所有 lane 的 bit mask。
    //
    // lane0:
    // 000000...
    //
    // lane1:
    // 000001
    //
    // lane2:
    // 000011
    //
    // ...
    unsigned lane_mask_lt =
        (1u << lane) - 1u;


    // 当前 valid 元素在 dense B 中的位置。
    int rank =
        __popc(
            valid_mask &
            lane_mask_lt);


    // V1 原来的 Shared Scatter。
    if (!duplicated) {

        s_compact[
            shared_base +
            rank
        ] =
            b;
    }


    __syncwarp(
        FULL_MASK);


    // 下面只是 benchmark 为了消费 compact 结果，
    // 防止编译器删除前面的工作。
    int compact_value =
        lane < count
            ? s_compact[
                  shared_base +
                  lane
              ]
            : 0;


    // Warp checksum。
#pragma unroll
    for (int offset = 16;
         offset > 0;
         offset >>= 1) {

        compact_value +=
            __shfl_down_sync(
                FULL_MASK,
                compact_value,
                offset,
                WARP_SIZE);
    }


    checksum =
        compact_value;


    // 防止下一个 B 覆盖当前 warp 的 shared 数据。
    __syncwarp(
        FULL_MASK);
}


// 初始化数据。
//
// A:
// 0,2,4,...,62 的 permutation。
//
// B:
// 偶数 lane -> 一定存在于 A
// 奇数 lane -> 一定不存在于 A
//
// 因此每个 B：
// unique count = 16
__global__
void init_input_kernel(
    int* __restrict__ input_a,
    int* __restrict__ input_b,
    size_t num_groups)
{
    size_t idx =
        static_cast<size_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;


    size_t total =
        num_groups *
        WARP_SIZE;


    if (idx >= total) {
        return;
    }


    size_t group_id =
        idx /
        WARP_SIZE;


    int lane =
        static_cast<int>(
            idx %
            WARP_SIZE);


    // A
    int a_perm =
        (lane * 17) &
        31;


    input_a[
        group_id *
            WARP_SIZE +
        lane
    ] =
        a_perm * 2;


    // B0 ... BN
#pragma unroll
    for (int b_id = 0;
         b_id < NUM_B_LISTS;
         ++b_id) {

        size_t b_idx =
            (group_id *
                 NUM_B_LISTS +
             b_id) *
                WARP_SIZE +
            lane;


        if ((lane & 1) == 0) {

            // 偶数 lane：
            // 一定能在 A 中找到。
            int a_lane =
                (lane * 7 +
                 b_id * 3) &
                31;


            int perm =
                (a_lane * 17) &
                31;


            input_b[b_idx] =
                perm * 2;

        } else {

            // 奇数 lane：
            // 一定不存在于 A。
            input_b[b_idx] =
                4096 +
                b_id * 128 +
                lane;
        }
    }
}


// 正式 V3 Kernel
//
// A:
// Load once
// ↓
// Bitonic Sort once
//
// B:
// Load
// ↓
// Fixed-32 Specialized Search
// ↓
// Ballot
// ↓
// Popc rank
// ↓
// Shared Compact
//
// V3 只替换 Search。
__global__
__launch_bounds__(BLOCK_SIZE)
void v3_fixed32_kernel(
    const int* __restrict__ input_a,
    const int* __restrict__ input_b,
    int* __restrict__ output_count,
    int* __restrict__ output_checksum)
{
    // 和 V1 完全一样：
    //
    // 16 warps/block
    // × 32 ints
    // × 4 bytes
    //
    // = 2048 bytes shared/block
    __shared__ int s_compact[BLOCK_SIZE];


    int tx =
        threadIdx.x;


    int warp_id =
        tx >> 5;


    int lane =
        tx & 31;


    size_t group_id =
        static_cast<size_t>(blockIdx.x) *
            NUM_WARPS +
        warp_id;


    int shared_base =
        warp_id *
        WARP_SIZE;


    // Step 1
    // Load A once。
    int a_value =
        input_a[
            group_id *
                WARP_SIZE +
            lane];


    // Step 2
    // Sort A once。
    int sorted_a =
        warp_bitonic_sort(
            a_value,
            lane);


    // Step 3
    // 每个 B 复用同一个 sorted A。
#pragma unroll
    for (int b_id = 0;
         b_id < NUM_B_LISTS;
         ++b_id) {


        size_t b_idx =
            (group_id *
                 NUM_B_LISTS +
             b_id) *
                WARP_SIZE +
            lane;


        // Load 当前 B。
        int b =
            input_b[b_idx];


        // V3 唯一核心变化：
        //
        // 不再：
        // left/right generic binary search
        //
        // 改成：
        // Fixed-32 specialized search
        bool duplicated =
            warp_fixed32_membership(
                sorted_a,
                b);


        // 后面的 compact 与 V1 相同。
        int count;
        int checksum;


        compact_and_consume(
            b,
            duplicated,
            lane,
            shared_base,
            s_compact,
            count,
            checksum);


        if (lane == 0) {

            size_t out_idx =
                group_id *
                    NUM_B_LISTS +
                b_id;


            output_count[
                out_idx
            ] =
                count;


            output_checksum[
                out_idx
            ] =
                checksum;
        }
    }
}


int main()
{
    int device = 0;


    CUDA_CHECK(
        cudaGetDevice(
            &device));


    cudaDeviceProp prop{};


    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            device));


    const size_t num_groups =
        static_cast<size_t>(NUM_BLOCKS) *
        NUM_WARPS;


    const size_t num_a_elements =
        num_groups *
        WARP_SIZE;


    const size_t num_b_elements =
        num_groups *
        NUM_B_LISTS *
        WARP_SIZE;


    const size_t num_outputs =
        num_groups *
        NUM_B_LISTS;


    const size_t a_bytes =
        num_a_elements *
        sizeof(int);


    const size_t b_bytes =
        num_b_elements *
        sizeof(int);


    const size_t output_bytes =
        num_outputs *
        sizeof(int);


    printf(
        "GPU             : %s\n",
        prop.name);


    printf(
        "Grid            : <<<%d, %d>>>\n",
        NUM_BLOCKS,
        BLOCK_SIZE);


    printf(
        "Warps/block     : %d\n",
        NUM_WARPS);


    printf(
        "Groups          : %zu\n",
        num_groups);


    printf(
        "NUM_B_LISTS     : %d\n",
        NUM_B_LISTS);


    printf(
        "A memory        : %.2f MiB\n",
        a_bytes /
            1024.0 /
            1024.0);


    printf(
        "B memory        : %.2f MiB\n",
        b_bytes /
            1024.0 /
            1024.0);


    int* d_a = nullptr;
    int* d_b = nullptr;

    int* d_count = nullptr;
    int* d_checksum = nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_a,
            a_bytes));


    CUDA_CHECK(
        cudaMalloc(
            &d_b,
            b_bytes));


    CUDA_CHECK(
        cudaMalloc(
            &d_count,
            output_bytes));


    CUDA_CHECK(
        cudaMalloc(
            &d_checksum,
            output_bytes));


    // 初始化输入。
    constexpr int INIT_THREADS =
        256;


    size_t init_total =
        num_groups *
        WARP_SIZE;


    int init_blocks =
        static_cast<int>(
            (init_total +
             INIT_THREADS - 1) /
            INIT_THREADS);


    init_input_kernel<<<
        init_blocks,
        INIT_THREADS>>>(
            d_a,
            d_b,
            num_groups);


    CUDA_CHECK(
        cudaGetLastError());


    CUDA_CHECK(
        cudaDeviceSynchronize());


    // Warmup。
    constexpr int WARMUP =
        10;


    for (int i = 0;
         i < WARMUP;
         ++i) {

        v3_fixed32_kernel<<<
            NUM_BLOCKS,
            BLOCK_SIZE>>>(
                d_a,
                d_b,
                d_count,
                d_checksum);
    }


    CUDA_CHECK(
        cudaGetLastError());


    CUDA_CHECK(
        cudaDeviceSynchronize());


    // Benchmark。
    constexpr int REPEAT =
        50;


    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(
        cudaEventCreate(
            &start));


    CUDA_CHECK(
        cudaEventCreate(
            &stop));


    CUDA_CHECK(
        cudaEventRecord(
            start));


    for (int i = 0;
         i < REPEAT;
         ++i) {

        v3_fixed32_kernel<<<
            NUM_BLOCKS,
            BLOCK_SIZE>>>(
                d_a,
                d_b,
                d_count,
                d_checksum);
    }


    CUDA_CHECK(
        cudaEventRecord(
            stop));


    CUDA_CHECK(
        cudaEventSynchronize(
            stop));


    float elapsed_ms =
        0.0f;


    CUDA_CHECK(
        cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop));


    float avg_ms =
        elapsed_ms /
        REPEAT;


    // Copy correctness output。
    std::vector<int>
        h_count(
            num_outputs);


    std::vector<int>
        h_checksum(
            num_outputs);


    CUDA_CHECK(
        cudaMemcpy(
            h_count.data(),
            d_count,
            output_bytes,
            cudaMemcpyDeviceToHost));


    CUDA_CHECK(
        cudaMemcpy(
            h_checksum.data(),
            d_checksum,
            output_bytes,
            cudaMemcpyDeviceToHost));


    // Correctness。
    size_t errors =
        0;


    for (size_t group = 0;
         group < num_groups;
         ++group) {

        for (int b_id = 0;
             b_id < NUM_B_LISTS;
             ++b_id) {

            size_t idx =
                group *
                    NUM_B_LISTS +
                b_id;


            // unique B 为奇数 lane：
            //
            // 1+3+...+31 = 256
            //
            // count = 16
            int expected_checksum =
                16 * 4096 +
                16 * b_id * 128 +
                256;


            if (h_count[idx] != 16 ||
                h_checksum[idx] !=
                    expected_checksum) {

                if (errors < 10) {

                    printf(
                        "Mismatch group=%zu b=%d "
                        "count=%d checksum=%d "
                        "expected_count=16 "
                        "expected_checksum=%d\n",
                        group,
                        b_id,
                        h_count[idx],
                        h_checksum[idx],
                        expected_checksum);
                }


                ++errors;
            }
        }
    }


    // 和 V1 / V2 使用完全相同的
    // Logical Global Traffic。
    //
    // A:
    // 128 B / group
    //
    // 每个 B:
    // B input      128 B
    // count          4 B
    // checksum       4 B
    //
    // = 136 B / B
    double bytes_per_group =
        128.0 +
        NUM_B_LISTS *
            136.0;


    double total_bytes =
        bytes_per_group *
        num_groups;


    double seconds =
        avg_ms /
        1000.0;


    double effective_bw =
        total_bytes /
        seconds /
        1e9;


    double groups_per_sec =
        num_groups /
        seconds;


    double pairs_per_sec =
        num_groups *
        NUM_B_LISTS /
        seconds;


    printf("\n");


    printf(
        "V3 1A-NB Sort Once + Fixed-32 Search\n");


    printf(
        "Average time    : %.6f ms\n",
        avg_ms);


    printf(
        "Groups/s        : %.3f M\n",
        groups_per_sec /
            1e6);


    printf(
        "A-B pairs/s     : %.3f M\n",
        pairs_per_sec /
            1e6);


    printf(
        "Logical traffic : %.3f GB\n",
        total_bytes /
            1e9);


    printf(
        "Effective BW    : %.3f GB/s\n",
        effective_bw);


    printf(
        "Search probes   : %d / group\n",
        15 +
            NUM_B_LISTS *
                6);


    printf(
        "Search type     : Fixed-32\n");


    printf(
        "Wrong results   : %zu\n",
        errors);


    CUDA_CHECK(
        cudaEventDestroy(
            start));


    CUDA_CHECK(
        cudaEventDestroy(
            stop));


    CUDA_CHECK(
        cudaFree(
            d_a));


    CUDA_CHECK(
        cudaFree(
            d_b));


    CUDA_CHECK(
        cudaFree(
            d_count));


    CUDA_CHECK(
        cudaFree(
            d_checksum));


    return 0;
}