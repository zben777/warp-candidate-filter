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
              "V7 cp.async version requires NUM_B_LISTS == 4.");

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

constexpr int TOTAL_GROUPS = 4 * 1024 * 1024;

constexpr int NUM_BLOCKS =
    TOTAL_GROUPS / NUM_WARPS;

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


// cp.async:
// Global -> Shared
//
// 每个线程搬 16 Bytes。
// V7 的 B layout 为：
// [group][lane][4]
//
// 所以每个线程自己的：
// B0 B1 B2 B3
// 正好连续 16B。
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
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr),
          "l"(gmem_ptr)
        : "memory");

#else

    // 这里只是为了让代码在低架构下仍有定义。
    // 当前实际编译目标 sm_89 不会走这里。
    *reinterpret_cast<int4*>(smem_ptr) =
        *reinterpret_cast<const int4*>(gmem_ptr);

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


// lane bit extract。
// 和 V5/V6 一致。
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


// A[32] Warp Bitonic Sort。
// 完全保持 V6。
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


// V6 Generic Binary Search。
// 不修改。
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


// V6 Shared Scatter Compact。
// 不修改。
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


// 输入布局与 V6 完全相同。
//
// A:
// [group][lane]
//
// B:
// [group][lane][B]
//
// 对一个 lane：
// B0 B1 B2 B3
// 连续 16 Bytes。
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


// V7:
//
// V6:
// Global B
//   -> LDG.E.128
//   -> Registers
//
// V7:
// Global B
//   -> cp.async 16B/thread
//   -> Shared
//
// 然后利用：
// A load + Bitonic Sort
//
// 覆盖 B 的 Global Memory latency。
__global__
__launch_bounds__(BLOCK_SIZE)
void v7_cp_async_b_kernel(
    const int* __restrict__ input_a,
    const int* __restrict__ input_b,
    int* __restrict__ output_count,
    int* __restrict__ output_checksum)
{
    // 原有 compact buffer:
    //
    // 512 int
    // = 2048B
    __shared__ int s_compact[
        BLOCK_SIZE];


    // V5/V6 output staging:
    //
    // 64 int + 64 int
    // = 512B
    __shared__ int s_output_count[
        RESULTS_PER_BLOCK];

    __shared__ int s_output_checksum[
        RESULTS_PER_BLOCK];


    // V7 新增：
    //
    // 每个 thread 一个 int4
    //
    // 512 threads × 16B
    // = 8192B
    //
    // 一个 warp：
    // 32 × int4
    // = 512B
    //
    // 正好对应该 group 的 B0/B1/B2/B3。
    __shared__ __align__(16)
    int4 s_b4[
        BLOCK_SIZE];


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


    // V6 B layout:
    //
    // [group][lane][4]
    //
    // 所以 reinterpret_cast<int4> 后：
    //
    // 一个 group 有 32 个 int4。
    const int4* __restrict__ input_b4 =
        reinterpret_cast<const int4*>(
            input_b);


    const int4* b_src =
        input_b4 +
        group_id *
            WARP_SIZE +
        lane;


    // V7 核心：
    //
    // 尽可能早地发起 B：
    //
    // Global -> Shared
    //
    // 每线程 16B。
    //
    // 此处并不等待。
    cp_async_16(
        &s_b4[tx],
        b_src);


    cp_async_commit();


    // 在 B 正在异步搬运时，
    // 开始处理 A。
    //
    // 注意：
    // 这里故意把 A load 放在 cp.async 后面，
    // 给 B 尽可能长的 overlap window。
    int a_value =
        input_a[
            group_id *
                WARP_SIZE +
            lane];


    // 15-stage Bitonic Sort。
    //
    // 理论目标：
    //
    // B cp.async
    //       │
    //       ├───────────────┐
    //       │               │
    //       │    Load A     │
    //       │    Sort A     │
    //       │               │
    //       └───────────────┘
    //
    // 用这些工作隐藏 B latency。
    int sorted_a =
        warp_bitonic_sort(
            a_value,
            lane);


    // 到真正需要 B 时才等待。
    cp_async_wait_all();


    // 每个 lane 都必须等自己的 async copy 完成。
    //
    // 然后 warp barrier 确保整个 warp
    // 可以安全读取 shared 中的数据。
    // __syncwarp(
    //     FULL_MASK);


    // 从 Shared 一次取回该线程的：
    //
    // B0 B1 B2 B3
    //
    // 注意：
    // 这里会增加一次 Shared Memory read，
    // 这是 V7 latency hiding 要付出的代价。
    int4 b_vec =
        s_b4[tx];


    // 后续算法完全保持 V6。
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


    // V5/V6 Block-Coalesced Output Store。
    __syncthreads();


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
        "B load          : cp.async 16B/thread\n");


    printf(
        "Prefetch target : Global -> Shared\n");


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

        v7_cp_async_b_kernel<<<
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

        v7_cp_async_b_kernel<<<
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


    // 和 V5/V6 相同的 logical traffic。
    //
    // A:
    // 128B
    //
    // B:
    // 512B
    //
    // count+checksum:
    // 32B
    //
    // total:
    // 672B/group
    //
    // 注意：
    // cp.async 并没有减少 DRAM bytes。
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
        "V7 1A-NB cp.async B Prefetch\n");


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
        "B transfer      : cp.async 16B/thread\n");


    printf(
        "Overlap         : B copy with A load + sort\n");


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