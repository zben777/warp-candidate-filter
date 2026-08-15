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


// ------------------------------------------------------------
// Bitonic Sort helpers
// 和 V1/V2/V3 保持一致
// ------------------------------------------------------------

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


// ------------------------------------------------------------
// Warp Bitonic Sort
//
// 输入：
// 每个 lane 持有一个 A 元素
//
// 输出：
// lane0  -> sorted_A[0]
// ...
// lane31 -> sorted_A[31]
// ------------------------------------------------------------

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


// ------------------------------------------------------------
// V4 核心：
// 2-way Fixed-32 Membership Search
//
// V2：
// 两套 generic binary search
//
//     left0/right0
//     left1/right1
//     active0/active1
//     mid0/mid1
//     existed0/existed1
//
// V4：
// 两条搜索链只维护：
//
//     target0 + pos0
//     target1 + pos1
//
// 固定 stride：
//
//     16
//      8
//      4
//      2
//      1
//
// 每一级：
// 先发 B0 和 B1 两个独立 SHFL，
// 再分别更新 pos0 / pos1。
// ------------------------------------------------------------

__device__ __forceinline__
void warp_fixed32_membership_2way(
    int sorted_a,
    int target0,
    int target1,
    bool& existed0,
    bool& existed1)
{
    int pos0 = -1;
    int pos1 = -1;


    // --------------------------------------------------------
    // stride = 16
    // --------------------------------------------------------

    {
        int probe0 = pos0 + 16;
        int probe1 = pos1 + 16;

        // 两个独立的 SHFL
        int value0 =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe0,
                WARP_SIZE);

        int value1 =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe1,
                WARP_SIZE);

        pos0 =
            (value0 < target0)
                ? probe0
                : pos0;

        pos1 =
            (value1 < target1)
                ? probe1
                : pos1;
    }


    // --------------------------------------------------------
    // stride = 8
    // --------------------------------------------------------

    {
        int probe0 = pos0 + 8;
        int probe1 = pos1 + 8;

        int value0 =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe0,
                WARP_SIZE);

        int value1 =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe1,
                WARP_SIZE);

        pos0 =
            (value0 < target0)
                ? probe0
                : pos0;

        pos1 =
            (value1 < target1)
                ? probe1
                : pos1;
    }


    // --------------------------------------------------------
    // stride = 4
    // --------------------------------------------------------

    {
        int probe0 = pos0 + 4;
        int probe1 = pos1 + 4;

        int value0 =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe0,
                WARP_SIZE);

        int value1 =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe1,
                WARP_SIZE);

        pos0 =
            (value0 < target0)
                ? probe0
                : pos0;

        pos1 =
            (value1 < target1)
                ? probe1
                : pos1;
    }


    // --------------------------------------------------------
    // stride = 2
    // --------------------------------------------------------

    {
        int probe0 = pos0 + 2;
        int probe1 = pos1 + 2;

        int value0 =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe0,
                WARP_SIZE);

        int value1 =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe1,
                WARP_SIZE);

        pos0 =
            (value0 < target0)
                ? probe0
                : pos0;

        pos1 =
            (value1 < target1)
                ? probe1
                : pos1;
    }


    // --------------------------------------------------------
    // stride = 1
    // --------------------------------------------------------

    {
        int probe0 = pos0 + 1;
        int probe1 = pos1 + 1;

        int value0 =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe0,
                WARP_SIZE);

        int value1 =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                probe1,
                WARP_SIZE);

        pos0 =
            (value0 < target0)
                ? probe0
                : pos0;

        pos1 =
            (value1 < target1)
                ? probe1
                : pos1;
    }


    // --------------------------------------------------------
    // 最终 candidate
    //
    // pos 是最后一个 < target 的位置
    //
    // candidate = pos + 1
    //
    // candidate 一定位于 [0,31]
    // --------------------------------------------------------

    int candidate0 =
        pos0 + 1;

    int candidate1 =
        pos1 + 1;


    // 仍然先连续发两个独立 SHFL。
    int final_value0 =
        __shfl_sync(
            FULL_MASK,
            sorted_a,
            candidate0,
            WARP_SIZE);

    int final_value1 =
        __shfl_sync(
            FULL_MASK,
            sorted_a,
            candidate1,
            WARP_SIZE);


    existed0 =
        (final_value0 == target0);

    existed1 =
        (final_value1 == target1);
}


// ------------------------------------------------------------
// 如果 NUM_B_LISTS 是奇数，
// 最后剩余一个 B 用单路 Fixed-32。
// 4B benchmark 不会走这里。
// ------------------------------------------------------------

__device__ __forceinline__
bool warp_fixed32_membership_1way(
    int sorted_a,
    int target)
{
    int pos = -1;


    {
        int probe = pos + 16;

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


    {
        int probe = pos + 8;

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


    {
        int probe = pos + 4;

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


    {
        int probe = pos + 2;

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


    {
        int probe = pos + 1;

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


// ------------------------------------------------------------
// Compact
//
// 完全沿用 V1。
// V4 不修改 Compact。
//
// duplicated
// ↓
// ballot
// ↓
// valid_mask
// ↓
// popc rank
// ↓
// Shared Scatter
// ↓
// dense B
// ------------------------------------------------------------

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


    // 当前 lane 左边所有 lane 的 mask。
    unsigned lane_mask_lt =
        (1u << lane) - 1u;


    // 当前 valid B 在 compact array 中的位置。
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


    // --------------------------------------------------------
    // 以下仅用于 microbenchmark：
    // 消费 compact 后的 B，
    // 防止 compiler DCE，同时验证结果。
    // --------------------------------------------------------

    int compact_value =
        lane < count
            ? s_compact[
                  shared_base +
                  lane
              ]
            : 0;


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


    // 当前 B 使用完 Shared 后，
    // 才能处理下一个 B。
    __syncwarp(
        FULL_MASK);
}


// ------------------------------------------------------------
// 初始化输入
//
// A:
// permutation of 0,2,4,...,62
//
// 每个 B：
// 偶数 lane -> 一定存在于 A
// 奇数 lane -> 一定不在 A
//
// 所以每个 B 最终 unique count = 16
// ------------------------------------------------------------

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


    // --------------------------------------------------------
    // A
    // --------------------------------------------------------

    int a_perm =
        (lane * 17) &
        31;


    input_a[
        group_id *
            WARP_SIZE +
        lane
    ] =
        a_perm * 2;


    // --------------------------------------------------------
    // B0 ... BN
    // --------------------------------------------------------

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

            // 一定在 A 中。
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

            // 一定不在 A 中。
            input_b[b_idx] =
                4096 +
                b_id * 128 +
                lane;
        }
    }
}


// ------------------------------------------------------------
// 正式 V4
//
// Sort A Once
// +
// Fixed-32 Search
// +
// 2-way B ILP
// +
// V1 Shared Scatter Compact
// ------------------------------------------------------------

__global__
__launch_bounds__(BLOCK_SIZE)
void v4_fixed32_2way_ilp_kernel(
    const int* __restrict__ input_a,
    const int* __restrict__ input_b,
    int* __restrict__ output_count,
    int* __restrict__ output_checksum)
{
    // 与 V1/V2/V3 一样：
    //
    // 512 int × 4B
    // = 2048 B Shared / block
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


    // --------------------------------------------------------
    // Step 1
    // Load A once
    // --------------------------------------------------------

    int a_value =
        input_a[
            group_id *
                WARP_SIZE +
            lane];


    // --------------------------------------------------------
    // Step 2
    // Sort A once
    //
    // sorted A 保留在 Warp Registers
    // --------------------------------------------------------

    int sorted_a =
        warp_bitonic_sort(
            a_value,
            lane);


    // --------------------------------------------------------
    // Step 3
    // 每次同时处理两个 B
    // --------------------------------------------------------

#pragma unroll
    for (int b_base = 0;
         b_base + 1 < NUM_B_LISTS;
         b_base += 2) {

        int b_id0 =
            b_base;

        int b_id1 =
            b_base + 1;


        size_t b_idx0 =
            (group_id *
                 NUM_B_LISTS +
             b_id0) *
                WARP_SIZE +
            lane;


        size_t b_idx1 =
            (group_id *
                 NUM_B_LISTS +
             b_id1) *
                WARP_SIZE +
            lane;


        // ----------------------------------------------------
        // 两个 B 同时 load 到 Register
        // ----------------------------------------------------

        int b0 =
            input_b[b_idx0];

        int b1 =
            input_b[b_idx1];


        bool duplicated0;
        bool duplicated1;


        // ----------------------------------------------------
        // V4 核心
        //
        // V3 Fixed-32
        // +
        // V2 2-way ILP
        // ----------------------------------------------------

        warp_fixed32_membership_2way(
            sorted_a,
            b0,
            b1,
            duplicated0,
            duplicated1);


        // ----------------------------------------------------
        // B0 Compact
        //
        // 完全使用 V1 方法
        // ----------------------------------------------------

        int count0;
        int checksum0;


        compact_and_consume(
            b0,
            duplicated0,
            lane,
            shared_base,
            s_compact,
            count0,
            checksum0);


        if (lane == 0) {

            size_t out_idx0 =
                group_id *
                    NUM_B_LISTS +
                b_id0;


            output_count[
                out_idx0
            ] =
                count0;


            output_checksum[
                out_idx0
            ] =
                checksum0;
        }


        // ----------------------------------------------------
        // B1 Compact
        // ----------------------------------------------------

        int count1;
        int checksum1;


        compact_and_consume(
            b1,
            duplicated1,
            lane,
            shared_base,
            s_compact,
            count1,
            checksum1);


        if (lane == 0) {

            size_t out_idx1 =
                group_id *
                    NUM_B_LISTS +
                b_id1;


            output_count[
                out_idx1
            ] =
                count1;


            output_checksum[
                out_idx1
            ] =
                checksum1;
        }
    }


    // --------------------------------------------------------
    // NUM_B_LISTS 如果是奇数：
    // 最后一个 B 单独处理。
    //
    // 当前 NUM_B_LISTS=4 不会进入。
    // --------------------------------------------------------

#if (NUM_B_LISTS % 2) == 1

    constexpr int b_id =
        NUM_B_LISTS - 1;


    size_t b_idx =
        (group_id *
             NUM_B_LISTS +
         b_id) *
            WARP_SIZE +
        lane;


    int b =
        input_b[b_idx];


    bool duplicated =
        warp_fixed32_membership_1way(
            sorted_a,
            b);


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

#endif
}


// ------------------------------------------------------------
// main
// ------------------------------------------------------------

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


    // --------------------------------------------------------
    // Init
    // --------------------------------------------------------

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


    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    constexpr int WARMUP =
        10;


    for (int i = 0;
         i < WARMUP;
         ++i) {

        v4_fixed32_2way_ilp_kernel<<<
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


    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

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

        v4_fixed32_2way_ilp_kernel<<<
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


    // --------------------------------------------------------
    // Copy Results
    // --------------------------------------------------------

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


    // --------------------------------------------------------
    // Correctness
    // --------------------------------------------------------

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


            // unique 元素：
            // odd lane = 1,3,...31
            //
            // 1+3+...+31 = 256
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


    // --------------------------------------------------------
    // Logical Traffic
    //
    // 和 V1/V2/V3 完全一样：
    //
    // A:
    // 128 B
    //
    // 每个 B:
    // input       128 B
    // count         4 B
    // checksum      4 B
    //
    // = 136 B / B
    //
    // NUM_B_LISTS=4:
    //
    // 128 + 4*136
    // = 672 B/group
    // --------------------------------------------------------

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


    // --------------------------------------------------------
    // Output
    // --------------------------------------------------------

    printf("\n");


    printf(
        "V4 1A-NB Fixed-32 + 2-Way B ILP\n");


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
        "B ILP width     : 2\n");


    printf(
        "Wrong results   : %zu\n",
        errors);


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

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