#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef NUM_B_LISTS
#define NUM_B_LISTS 4
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

// V8:
// 一个 warp 连续处理两个 group
constexpr int GROUPS_PER_WARP = 2;

constexpr int TOTAL_GROUPS =
    4 * 1024 * 1024;

constexpr int GROUPS_PER_BLOCK =
    NUM_WARPS * GROUPS_PER_WARP;

constexpr int NUM_BLOCKS =
    TOTAL_GROUPS / GROUPS_PER_BLOCK;

constexpr int RESULTS_PER_GROUP =
    NUM_B_LISTS;

constexpr int RESULTS_PER_WARP =
    GROUPS_PER_WARP * NUM_B_LISTS;

constexpr int RESULTS_PER_BLOCK =
    GROUPS_PER_BLOCK * NUM_B_LISTS;

constexpr unsigned FULL_MASK =
    0xffffffffu;

static_assert(
    NUM_B_LISTS == 4,
    "V8 requires NUM_B_LISTS == 4.");

static_assert(
    TOTAL_GROUPS % GROUPS_PER_BLOCK == 0,
    "TOTAL_GROUPS must be divisible by GROUPS_PER_BLOCK.");

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr,                                                    \
                "CUDA error %s:%d: %s\n",                                  \
                __FILE__,                                                  \
                __LINE__,                                                  \
                cudaGetErrorString(err));                                  \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)


// ------------------------------------------------------------
// cp.async helpers
// ------------------------------------------------------------

// B:
//
// 每线程搬一个 int4
// = 16 Bytes
//
// 使用 .cg：
// Global -> Shared
// 16B copy。
__device__ __forceinline__
void cp_async_16(
    void* smem_ptr,
    const void* gmem_ptr)
{
#if __CUDA_ARCH__ >= 800

    unsigned smem_addr =
        static_cast<unsigned>(
            __cvta_generic_to_shared(
                smem_ptr));

    asm volatile(
        "cp.async.cg.shared.global "
        "[%0], [%1], 16;\n"
        :
        : "r"(smem_addr),
          "l"(gmem_ptr)
        : "memory");

#else

    *reinterpret_cast<int4*>(smem_ptr) =
        *reinterpret_cast<const int4*>(gmem_ptr);

#endif
}


// A:
//
// 每线程只有一个 int
// = 4 Bytes
//
// 这里使用 .ca。
// cp.async.ca 支持 4B copy。
__device__ __forceinline__
void cp_async_4(
    void* smem_ptr,
    const void* gmem_ptr)
{
#if __CUDA_ARCH__ >= 800

    unsigned smem_addr =
        static_cast<unsigned>(
            __cvta_generic_to_shared(
                smem_ptr));

    asm volatile(
        "cp.async.ca.shared.global "
        "[%0], [%1], 4;\n"
        :
        : "r"(smem_addr),
          "l"(gmem_ptr)
        : "memory");

#else

    *reinterpret_cast<int*>(smem_ptr) =
        *reinterpret_cast<const int*>(gmem_ptr);

#endif
}


__device__ __forceinline__
void cp_async_commit()
{
#if __CUDA_ARCH__ >= 800

    asm volatile(
        "cp.async.commit_group;\n"
        :
        :
        : "memory");

#endif
}


__device__ __forceinline__
void cp_async_wait_all()
{
#if __CUDA_ARCH__ >= 800

    asm volatile(
        "cp.async.wait_group 0;\n"
        :
        :
        : "memory");

#endif
}


// ------------------------------------------------------------
// Warp Bitonic Sort
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
// Generic Binary Search
// ------------------------------------------------------------

__device__ __forceinline__
bool warp_binary_search(
    int sorted_a,
    int target)
{
    int left =
        0;

    int right =
        WARP_SIZE;

    bool existed =
        false;

#pragma unroll
    for (int step = 0;
         step < 6;
         ++step) {

        bool active =
            (!existed &&
             left < right);

        int mid =
            active
                ? ((left + right) >> 1)
                : 0;

        int value =
            __shfl_sync(
                FULL_MASK,
                sorted_a,
                mid,
                WARP_SIZE);

        if (active) {

            if (value == target) {

                existed =
                    true;

            } else if (target > value) {

                left =
                    mid + 1;

            } else {

                right =
                    mid;
            }
        }
    }

    return existed;
}


// ------------------------------------------------------------
// Shared Scatter Compact
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


    unsigned lane_mask_lt =
        (1u << lane) - 1u;


    int rank =
        __popc(
            valid_mask &
            lane_mask_lt);


    if (!duplicated) {

        s_compact[
            shared_base +
            rank
        ] =
            b;
    }


    __syncwarp(
        FULL_MASK);


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


    __syncwarp(
        FULL_MASK);
}


// ------------------------------------------------------------
// 处理一个已经进入 Register 的 Group
//
// a_value:
// 当前 lane 的 A
//
// b_vec:
// 当前 lane 的 B0/B1/B2/B3
// ------------------------------------------------------------

__device__ __forceinline__
void process_group(
    int a_value,
    int4 b_vec,
    int lane,
    int shared_base,
    int local_result_base,
    int* s_compact,
    int* s_output_count,
    int* s_output_checksum)
{
    // A sort once
    int sorted_a =
        warp_bitonic_sort(
            a_value,
            lane);


#pragma unroll
    for (int b_id = 0;
         b_id < NUM_B_LISTS;
         ++b_id) {

        int b;


        if (b_id == 0) {

            b =
                b_vec.x;

        } else if (b_id == 1) {

            b =
                b_vec.y;

        } else if (b_id == 2) {

            b =
                b_vec.z;

        } else {

            b =
                b_vec.w;
        }


        bool duplicated =
            warp_binary_search(
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

            int result_idx =
                local_result_base +
                b_id;


            s_output_count[
                result_idx
            ] =
                count;


            s_output_checksum[
                result_idx
            ] =
                checksum;
        }
    }
}


// ------------------------------------------------------------
// Input initialization
//
// A:
// [group][lane]
//
// B:
// [group][lane][B]
//
// 保持和 V6/V7 完全相同。
// ------------------------------------------------------------

__global__
void init_input_kernel(
    int* __restrict__ input_a,
    int* __restrict__ input_b,
    size_t num_groups)
{
    size_t idx =
        static_cast<size_t>(
            blockIdx.x) *
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


    int a_perm =
        (lane * 17) &
        31;


    input_a[
        group_id *
            WARP_SIZE +
        lane
    ] =
        a_perm * 2;


#pragma unroll
    for (int b_id = 0;
         b_id < NUM_B_LISTS;
         ++b_id) {

        size_t b_idx =
            (group_id *
                 WARP_SIZE +
             lane) *
                NUM_B_LISTS +
            b_id;


        if ((lane & 1) == 0) {

            int a_lane =
                (lane * 7 +
                 b_id * 3) &
                31;


            int perm =
                (a_lane * 17) &
                31;


            input_b[
                b_idx
            ] =
                perm * 2;

        } else {

            input_b[
                b_idx
            ] =
                4096 +
                b_id * 128 +
                lane;
        }
    }
}


// ------------------------------------------------------------
// V8
//
// 一个 warp:
// Group0 + Group1
//
// 关键 pipeline:
//
// Load Group0 -> Register
//
// cp.async Group1 A+B -> Shared
//
// Process Group0
//   Sort A0
//   Search B0 x4
//   Compact x4
//
// 此时 Group1 在后台搬运
//
// wait
//
// Group1 Shared -> Register
//
// Process Group1
//
// ------------------------------------------------------------

__global__
__launch_bounds__(BLOCK_SIZE)
void v8_two_group_pipeline_kernel(
    const int* __restrict__ input_a,
    const int* __restrict__ input_b,
    int* __restrict__ output_count,
    int* __restrict__ output_checksum)
{
    // 每个 warp 一个 32-int compact buffer
    //
    // 512 ints
    // = 2048B
    __shared__ int s_compact[
        BLOCK_SIZE];


    // V8:
    //
    // 一个 block：
    // 16 warps
    // × 2 groups
    // × 4 results
    // = 128 results
    //
    // count:
    // 128 × 4B = 512B
    //
    // checksum:
    // 128 × 4B = 512B
    __shared__ int s_output_count[
        RESULTS_PER_BLOCK];

    __shared__ int s_output_checksum[
        RESULTS_PER_BLOCK];


    // Next Group A staging
    //
    // 512 × 4B
    // = 2048B
    __shared__ int s_next_a[
        BLOCK_SIZE];


    // Next Group B staging
    //
    // 每 thread:
    // int4 = 16B
    //
    // 512 × 16B
    // = 8192B
    __shared__ __align__(16)
    int4 s_next_b4[
        BLOCK_SIZE];


    int tx =
        threadIdx.x;


    int warp_id =
        tx >> 5;


    int lane =
        tx &
        31;


    // 一个 block 处理：
    //
    // 16 warps × 2 groups
    // = 32 groups
    size_t block_group_base =
        static_cast<size_t>(
            blockIdx.x) *
        GROUPS_PER_BLOCK;


    // 每个 warp 连续处理两个 group
    size_t group0 =
        block_group_base +
        warp_id *
            GROUPS_PER_WARP;


    size_t group1 =
        group0 + 1;


    int shared_base =
        warp_id *
        WARP_SIZE;


    // block 内结果顺序：
    //
    // warp0:
    // G0 -> result 0..3
    // G1 -> result 4..7
    //
    // warp1:
    // G2 -> result 8..11
    // G3 -> result 12..15
    //
    // ...
    int warp_result_base =
        warp_id *
        RESULTS_PER_WARP;


    int group0_result_base =
        warp_result_base;


    int group1_result_base =
        warp_result_base +
        RESULTS_PER_GROUP;


    const int4* __restrict__ input_b4 =
        reinterpret_cast<const int4*>(
            input_b);


    // --------------------------------------------------------
    // Stage 1:
    // Group0 直接进入 Register
    // --------------------------------------------------------

    int a0 =
        input_a[
            group0 *
                WARP_SIZE +
            lane];


    int4 b0 =
        input_b4[
            group0 *
                WARP_SIZE +
            lane];


    // --------------------------------------------------------
    // Stage 2:
    // 提前异步搬 Group1
    //
    // A1: Global -> Shared
    // B1: Global -> Shared
    //
    // 两个 copy 放在同一个 async group
    // --------------------------------------------------------

    const int* next_a_src =
        input_a +
        group1 *
            WARP_SIZE +
        lane;


    const int4* next_b_src =
        input_b4 +
        group1 *
            WARP_SIZE +
        lane;


    cp_async_4(
        &s_next_a[tx],
        next_a_src);


    cp_async_16(
        &s_next_b4[tx],
        next_b_src);


    cp_async_commit();


    // --------------------------------------------------------
    // Stage 3:
    // Group1 在搬运的同时，
    // 完整处理 Group0。
    //
    // overlap window:
    //
    // Bitonic Sort
    // +
    // 4 × Binary Search
    // +
    // 4 × Compact
    // +
    // checksum reduction
    // --------------------------------------------------------

    process_group(
        a0,
        b0,
        lane,
        shared_base,
        group0_result_base,
        s_compact,
        s_output_count,
        s_output_checksum);


    // --------------------------------------------------------
    // Stage 4:
    // Group0 完成以后，
    // 才真正等待 Group1。
    // --------------------------------------------------------

    cp_async_wait_all();


    // 每个 thread 只读取
    // 自己 async-copy 的 Shared slot。
    //
    // 所以不需要额外 __syncwarp()。
    int a1 =
        s_next_a[
            tx];


    int4 b1 =
        s_next_b4[
            tx];


    // --------------------------------------------------------
    // Stage 5:
    // Group1 已经进入 Register。
    //
    // 现在 Shared staging buffer 生命周期结束。
    //
    // 然后正常处理 Group1。
    // --------------------------------------------------------

    process_group(
        a1,
        b1,
        lane,
        shared_base,
        group1_result_base,
        s_compact,
        s_output_count,
        s_output_checksum);


    // --------------------------------------------------------
    // Block-Coalesced Output
    //
    // 128 results/block
    //
    // thread0..127 连续输出
    // --------------------------------------------------------

    __syncthreads();


    if (tx < RESULTS_PER_BLOCK) {

        size_t global_result_idx =
            static_cast<size_t>(
                blockIdx.x) *
                RESULTS_PER_BLOCK +
            tx;


        output_count[
            global_result_idx
        ] =
            s_output_count[
                tx];


        output_checksum[
            global_result_idx
        ] =
            s_output_checksum[
                tx];
    }
}


// ------------------------------------------------------------
// main
// ------------------------------------------------------------

int main()
{
    int device =
        0;


    CUDA_CHECK(
        cudaGetDevice(
            &device));


    cudaDeviceProp prop{};


    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            device));


    const size_t num_groups =
        static_cast<size_t>(
            TOTAL_GROUPS);


    const size_t num_a_elements =
        num_groups *
        WARP_SIZE;


    const size_t num_b_elements =
        num_groups *
        WARP_SIZE *
        NUM_B_LISTS;


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
        "Groups/warp     : %d\n",
        GROUPS_PER_WARP);


    printf(
        "Groups/block    : %d\n",
        GROUPS_PER_BLOCK);


    printf(
        "Groups          : %zu\n",
        num_groups);


    printf(
        "NUM_B_LISTS     : %d\n",
        NUM_B_LISTS);


    printf(
        "Results/block   : %d\n",
        RESULTS_PER_BLOCK);


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


    printf(
        "B layout        : [group][lane][B]\n");


    printf(
        "Pipeline        : current compute + next A/B cp.async\n");


    int* d_a =
        nullptr;

    int* d_b =
        nullptr;

    int* d_count =
        nullptr;

    int* d_checksum =
        nullptr;


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

        v8_two_group_pipeline_kernel<<<
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

        v8_two_group_pipeline_kernel<<<
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
    // Correctness
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


            int expected_checksum =
                16 * 4096 +
                16 * b_id * 128 +
                256;


            if (h_count[idx] != 16 ||
                h_checksum[idx] !=
                    expected_checksum) {

                if (errors < 10) {

                    printf(
                        "Mismatch group=%zu "
                        "b=%d "
                        "count=%d "
                        "checksum=%d "
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
    // Logical traffic
    //
    // workload 完全没变：
    //
    // A:
    // 128B/group
    //
    // B:
    // 512B/group
    //
    // output:
    // 32B/group
    //
    // total:
    // 672B/group
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


    printf("\n");


    printf(
        "V8 1A-NB Two-Group cp.async Pipeline\n");


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
        "Groups/warp     : %d\n",
        GROUPS_PER_WARP);


    printf(
        "Search/group    : 15 + 4*6 stages\n");


    printf(
        "Search type     : Generic Binary Search\n");


    printf(
        "Output store    : Block-Coalesced\n");


    printf(
        "Next A transfer : cp.async 4B/thread\n");


    printf(
        "Next B transfer : cp.async 16B/thread\n");


    printf(
        "Overlap         : next group memory with current full compute\n");


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