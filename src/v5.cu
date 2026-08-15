#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef NUM_B_LISTS
#define NUM_B_LISTS 4
#endif

constexpr int WARP_SIZE = 32;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

// 总 group 数固定不变：4194304
constexpr int TOTAL_GROUPS = 4 * 1024 * 1024;

// 根据 block size 自动调整 grid size
constexpr int NUM_BLOCKS =
    TOTAL_GROUPS / NUM_WARPS;

// 每个 block 最终产生多少组 count/checksum
constexpr int RESULTS_PER_BLOCK =
    NUM_WARPS * NUM_B_LISTS;

constexpr unsigned FULL_MASK = 0xffffffffu;

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)


// 取 lane_id 的某一位。
// 保持和 V1 相同的 Bitonic Sort。
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


// Bitonic compare-exchange。
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


// A[32] Warp Bitonic Sort。
// 排序后：
// lane0  -> sorted_A[0]
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


// V1 原来的 Generic Binary Search。
// V5 完全不修改这里。
//
// 每个 lane 有自己的 target。
// sorted_a 分布在整个 warp 的 register。
__device__ __forceinline__
bool warp_binary_search(
    int sorted_a,
    int target)
{
    int left  = 0;
    int right = WARP_SIZE;

    bool existed = false;

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


// V1 原来的 Compact。
// V5 不修改这里。
//
// duplicated
// -> ballot
// -> popc(rank)
// -> Shared Scatter
// -> dense B
//
// checksum 仅用于 benchmark 消费 compact 结果。
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


    // 防止下一个 B 覆盖当前 warp 的 compact buffer。
    __syncwarp(
        FULL_MASK);
}


// 初始化输入。
// A 为 0,2,...,62 的 permutation。
//
// 每个 B：
// 偶数 lane -> 一定在 A 中
// 奇数 lane -> 一定不在 A 中
//
// 所以每个 B 最终 count = 16。
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
                 NUM_B_LISTS +
             b_id) *
                WARP_SIZE +
            lane;


        if ((lane & 1) == 0) {

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

            input_b[b_idx] =
                4096 +
                b_id * 128 +
                lane;
        }
    }
}


// V5:
//
// 前面完全等于 V1。
//
// 唯一变化：
//
// V1:
// warp lane0
// -> scattered global store
//
// V5:
// warp lane0
// -> block shared output buffer
// -> __syncthreads()
// -> thread0..RESULTS_PER_BLOCK-1
// -> coalesced global store
__global__
__launch_bounds__(BLOCK_SIZE)
void v5_block_coalesced_store_kernel(
    const int* __restrict__ input_a,
    const int* __restrict__ input_b,
    int* __restrict__ output_count,
    int* __restrict__ output_checksum)
{
    // 原 V1 compact buffer：
    //
    // 512 int
    // = 2048 B
    __shared__ int s_compact[BLOCK_SIZE];


    // V5 新增：
    //
    // 每个 block：
    // 16 warps × 4 B
    // = 64 results
    //
    // count:
    // 64 × 4B = 256B
    //
    // checksum:
    // 64 × 4B = 256B
    //
    // V5 总 static shared：
    // 2048 + 256 + 256
    // = 2560B（NUM_B_LISTS=4）
    __shared__ int s_output_count[
        RESULTS_PER_BLOCK];

    __shared__ int s_output_checksum[
        RESULTS_PER_BLOCK];


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


    // 当前 warp 的结果在 block output buffer 中的位置。
    //
    // warp0:
    // index 0,1,2,3
    //
    // warp1:
    // index 4,5,6,7
    //
    // ...
    int block_result_base =
        warp_id *
        NUM_B_LISTS;


    // Load A once。
    int a_value =
        input_a[
            group_id *
                WARP_SIZE +
            lane];


    // Sort A once。
    int sorted_a =
        warp_bitonic_sort(
            a_value,
            lane);


    // 每个 warp 处理 B0...BN。
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


        // Load B。
        int b =
            input_b[b_idx];


        // V1 Generic Binary Search。
        bool duplicated =
            warp_binary_search(
                sorted_a,
                b);


        // V1 Shared Scatter Compact。
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


        // V5 的第一个变化：
        //
        // 不再直接写 Global。
        //
        // 每个 warp 的 lane0
        // 把结果先写到 block shared buffer。
        if (lane == 0) {

            int local_result_idx =
                block_result_base +
                b_id;


            s_output_count[
                local_result_idx
            ] =
                count;


            s_output_checksum[
                local_result_idx
            ] =
                checksum;
        }
    }


    // 所有 warp 都必须先完成自己的 B0...BN。
    //
    // 因为下面要由其他 thread
    // 读取各个 warp 写进去的 shared output。
    __syncthreads();


    // V5 核心：
    //
    // 原来：
    //
    // warp0 lane0 写 result0
    // warp1 lane0 写 result4
    // ...
    //
    // 现在：
    //
    // thread0 写 result0
    // thread1 写 result1
    // thread2 写 result2
    // ...
    //
    // NUM_B_LISTS=4 时：
    // thread0~63 一起输出 64 个连续结果。
    if (tx < RESULTS_PER_BLOCK) {

        size_t global_result_idx =
            static_cast<size_t>(blockIdx.x) *
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


    // 初始化。
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

        v5_block_coalesced_store_kernel<<<
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

        v5_block_coalesced_store_kernel<<<
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


            // unique B:
            // lane1,3,...31
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


    // Logical Global Traffic 不变。
    //
    // 注意：
    // V5 优化的是 transaction/coalescing，
    // 不是减少 logical bytes。
    //
    // A:
    // 128B
    //
    // 每个 B:
    // input     128B
    // count       4B
    // checksum    4B
    //
    // NUM_B_LISTS=4:
    //
    // 128 + 4*136
    // = 672B/group
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
        "V5 1A-NB V1 + Block-Coalesced Output Store\n");


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
        "Search stages   : %d / group\n",
        15 +
            NUM_B_LISTS *
                6);


    printf(
        "Search type     : Generic Binary Search\n");


    printf(
        "Output store    : Block-Coalesced\n");


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