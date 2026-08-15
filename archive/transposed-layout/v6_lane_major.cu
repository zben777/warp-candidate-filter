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

static_assert(NUM_B_LISTS == 4,
              "V6 int4 version requires NUM_B_LISTS == 4.");

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

// 总 group 数保持和 V5 完全一致
constexpr int TOTAL_GROUPS = 4 * 1024 * 1024;

constexpr int NUM_BLOCKS =
    TOTAL_GROUPS / NUM_WARPS;

// 每个 block 输出：
// NUM_WARPS * 4 个 B 的结果
constexpr int RESULTS_PER_BLOCK =
    NUM_WARPS * NUM_B_LISTS;

constexpr unsigned FULL_MASK = 0xffffffffu;

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


// lane_id 指定位提取。
// 和 V5 保持一致。
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


// Warp Bitonic Sort。
// A[32] 排序之后，每个 lane 保存一个 sorted A。
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


// V5 的 Generic Binary Search。
// 完全不修改搜索算法。
//
// 每个 lane 有自己的 target。
// sorted_a 分布在 warp 32 个 lane 的 register 中。
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


// V5 原来的 Shared Scatter Compact。
// 不修改。
//
// valid
// -> ballot
// -> popc rank
// -> shared scatter
// -> dense compact result
// -> checksum consume
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
        __popc(valid_mask);


    // mask of lanes < current lane
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


    __syncwarp(FULL_MASK);


    int compact_value =
        lane < count
            ? s_compact[
                  shared_base +
                  lane
              ]
            : 0;


    // checksum 仅用于 benchmark：
    // 确保 compact 后的数据真的被消费，
    // 防止编译器把整个 compact 消掉。
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


    __syncwarp(FULL_MASK);
}


// V6 输入初始化。
//
// A layout 仍然：
// [group][lane]
//
// B layout 从 V5：
// [group][b_id][lane]
//
// 改成 V6：
// [group][lane][b_id]
//
// 因此一个 lane 的 B0/B1/B2/B3
// 在内存中连续排列。
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


    // A:
    // permutation of:
    //
    // 0,2,4,...,62
    int a_perm =
        (lane * 17) &
        31;


    input_a[
        group_id *
            WARP_SIZE +
        lane
    ] =
        a_perm * 2;


    // V6 B layout:
    //
    // [group][lane][B]
    //
    // 一个线程自己的：
    //
    // B0
    // B1
    // B2
    // B3
    //
    // 连续 16 Bytes
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

            // even lane：
            // 生成一个一定存在于 A 中的值
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

            // odd lane：
            // 一定不存在于 A 中
            input_b[b_idx] =
                4096 +
                b_id * 128 +
                lane;
        }
    }
}


// V6:
//
// V5:
// 每处理一个 B：
// scalar load B
// -> search
// -> compact
//
// V6:
// kernel 开始时：
// 一次 int4 load
// -> 得到 B0/B1/B2/B3
//
// 然后仍然逐个：
// search
// -> compact
//
// 注意：
// 没有使用 2-way ILP。
// 我们只测试 vectorized B load。
__global__
__launch_bounds__(BLOCK_SIZE)
void v6_int4_bload_kernel(
    const int* __restrict__ input_a,
    const int* __restrict__ input_b,
    int* __restrict__ output_count,
    int* __restrict__ output_checksum)
{
    // V5 Shared Compact Buffer
    __shared__ int s_compact[
        BLOCK_SIZE];


    // V5 Block-Coalesced Output Buffer
    __shared__ int s_output_count[
        RESULTS_PER_BLOCK];

    __shared__ int s_output_checksum[
        RESULTS_PER_BLOCK];


    int tx =
        threadIdx.x;


    int warp_id =
        tx >> 5;


    int lane =
        tx &
        31;


    size_t group_id =
        static_cast<size_t>(blockIdx.x) *
            NUM_WARPS +
        warp_id;


    int shared_base =
        warp_id *
        WARP_SIZE;


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


    // V6 核心变化
    //
    // B layout:
    //
    // [group][lane][4]
    //
    // 每个：
    //
    // group/lane
    //
    // 正好对应连续的：
    //
    // B0 B1 B2 B3
    //
    // 4 × int = 16B
    //
    // 所以可以 reinterpret_cast<int4>
    const int4* __restrict__ input_b4 =
        reinterpret_cast<const int4*>(
            input_b);


    // 每个 thread 一次 128-bit vector load。
    //
    // 理想 SASS/PTX 应该表现为
    // vectorized / 128-bit global load，
    // 后续可以通过 cuobjdump / nvdisasm 验证。
    int4 b_vec =
        input_b4[
            group_id *
                WARP_SIZE +
            lane];


    // 这里故意不搞额外 ILP。
    //
    // 每一个 B 仍然：
    //
    // search
    // -> compact
    //
    // 保持和 V5 的执行语义一致。
#pragma unroll
    for (int b_id = 0;
         b_id < NUM_B_LISTS;
         ++b_id) {

        int b;


        // 因为 NUM_B_LISTS == 4，
        // 编译器在 unroll 后可以直接选择
        // b_vec.x/y/z/w。
        if (b_id == 0) {

            b = b_vec.x;

        } else if (b_id == 1) {

            b = b_vec.y;

        } else if (b_id == 2) {

            b = b_vec.z;

        } else {

            b = b_vec.w;
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


        // V5 Block-Coalesced Store 的第一阶段：
        // 每个 warp lane0 先写 Shared。
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


    // 和 V5 一样：
    // 整个 block 汇总输出。
    __syncthreads();


    // thread0..63 连续写 global
    // （BLOCK_SIZE=512, NUM_B_LISTS=4）。
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
        "B load          : int4 / 128-bit per thread\n");


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

        v6_int4_bload_kernel<<<
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

        v6_int4_bload_kernel<<<
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


    // Copy outputs for correctness check。
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


    // 每个 B：
    //
    // even lane = duplicate
    // odd lane  = unique
    //
    // 所以 count = 16。
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


            // odd lane:
            //
            // 1 + 3 + ... + 31 = 256
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


    // Logical traffic 和 V5 完全一样。
    //
    // A:
    // 32 int = 128B
    //
    // B:
    // 4 × 32 int
    // = 512B
    //
    // output:
    // 4 × (count + checksum)
    // = 4 × 8B
    // = 32B
    //
    // total:
    // 672B / group
    //
    // 注意：
    // int4 并没有减少 DRAM bytes。
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
        "V6 1A-NB V5 + int4 B Load\n");


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
        "B layout        : [group][lane][B]\n");


    printf(
        "B load          : int4\n");


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