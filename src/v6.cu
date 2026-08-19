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

#ifndef SUBWARP_USE_CP_ASYNC
#define SUBWARP_USE_CP_ASYNC 0
#endif

#ifndef SUBWARP_FORCE_EARLY_CP_ASYNC
#define SUBWARP_FORCE_EARLY_CP_ASYNC 0
#endif

#ifndef SUBWARP_KERNEL
#define SUBWARP_KERNEL v6_subwarp_int4_kernel
#endif

#ifndef SUBWARP_VARIANT_LABEL
#define SUBWARP_VARIANT_LABEL "V6 4x8 Subwarp int4 Candidate Filter"
#endif

#ifndef SUBWARP_OVERLAP_LABEL
#define SUBWARP_OVERLAP_LABEL "compiler-scheduled async window"
#endif

constexpr int WARP_SIZE = 32;
constexpr int SUBWARP_SIZE = 8;
constexpr int VALUES_PER_THREAD = 4;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr int TOTAL_GROUPS = 4 * 1024 * 1024;
constexpr int NUM_BLOCKS = TOTAL_GROUPS / NUM_WARPS;
constexpr int RESULTS_PER_BLOCK = NUM_WARPS * NUM_B_LISTS;
constexpr int COMPACT_STRIDE = WARP_SIZE + 1;
constexpr unsigned FULL_MASK = 0xffffffffu;

static_assert(NUM_B_LISTS == 4, "V6 requires four B lists");
static_assert(BLOCK_SIZE % WARP_SIZE == 0, "Invalid block size");
static_assert(!SUBWARP_FORCE_EARLY_CP_ASYNC || SUBWARP_USE_CP_ASYNC,
              "Forced async scheduling requires cp.async");

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#if SUBWARP_USE_CP_ASYNC
__device__ __forceinline__ void cp_async_16(void* shared_ptr,
                                            const void* global_ptr)
{
    unsigned shared_address =
        static_cast<unsigned>(__cvta_generic_to_shared(shared_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(shared_address), "l"(global_ptr)
                 : "memory");
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" : : : "memory");
}

__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_group 0;\n" : : : "memory");
}
#endif

__device__ __forceinline__ unsigned bfe(unsigned lane, unsigned pos)
{
    unsigned result;
    asm("bfe.u32 %0,%1,%2,%3;" : "=r"(result) : "r"(lane), "r"(pos), "r"(1));
    return result;
}

__device__ __forceinline__ int xor_swap(int x, int mask, int dir)
{
    int y = __shfl_xor_sync(FULL_MASK, x, mask, WARP_SIZE);
    return ((x < y) == dir) ? y : x;
}

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

__device__ __forceinline__ bool warp_binary_search(int sorted_a, int target)
{
    int left = 0;
    int right = WARP_SIZE;
    bool existed = false;

#pragma unroll
    for (int step = 0; step < 6; ++step) {
        bool active = !existed && left < right;
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

__device__ __forceinline__ int component(const int4& values, int index)
{
    if (index == 0) return values.x;
    if (index == 1) return values.y;
    if (index == 2) return values.z;
    return values.w;
}

__global__ void init_input_kernel(int* __restrict__ input_a,
                                  int* __restrict__ input_b, size_t num_groups)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = num_groups * WARP_SIZE;
    if (idx >= total) return;

    size_t group_id = idx / WARP_SIZE;
    int lane = static_cast<int>(idx % WARP_SIZE);
    int a_perm = (lane * 17) & 31;
    input_a[group_id * WARP_SIZE + lane] = a_perm * 2;

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

// One warp handles one group. Four 8-lane subwarps handle B0..B3.
// Each lane loads four adjacent values from its subwarp's B list.
__global__ __launch_bounds__(BLOCK_SIZE) void SUBWARP_KERNEL(
    const int* __restrict__ input_a, const int* __restrict__ input_b,
    int* __restrict__ output_count, int* __restrict__ output_checksum)
{
    __shared__ int s_compact[NUM_WARPS * NUM_B_LISTS * COMPACT_STRIDE];
    __shared__ int s_output_count[RESULTS_PER_BLOCK];
    __shared__ int s_output_checksum[RESULTS_PER_BLOCK];
#if SUBWARP_USE_CP_ASYNC
    __shared__ __align__(16) int4 s_b4[BLOCK_SIZE];
#endif
#if SUBWARP_FORCE_EARLY_CP_ASYNC
    __shared__ volatile int s_schedule_a[BLOCK_SIZE];
#endif

    int tx = threadIdx.x;
    int warp_id = tx >> 5;
    int lane = tx & 31;
    int b_id = lane >> 3;
    int sub_lane = lane & 7;
    unsigned subwarp_mask = 0xffu << (b_id * SUBWARP_SIZE);

    size_t group_id = static_cast<size_t>(blockIdx.x) * NUM_WARPS + warp_id;

    // V5 layout: [group][B][element]. Eight adjacent int4 loads cover
    // one 128-byte B list; all 32 lanes cover the four lists contiguously.
    const int4* __restrict__ input_b4 = reinterpret_cast<const int4*>(input_b);
    const int4* b_source = input_b4 + group_id * WARP_SIZE + lane;

#if SUBWARP_USE_CP_ASYNC && SUBWARP_FORCE_EARLY_CP_ASYNC
    // Issue A, then B, before any operation consumes A. The warp ordering
    // point prevents ptxas from sinking LDGSTS to the end of the sort; it does
    // not wait for the asynchronous copy to complete.
    int a_value = input_a[group_id * WARP_SIZE + lane];
    s_schedule_a[tx] = a_value;
    cp_async_16(&s_b4[tx], b_source);
    cp_async_commit();
    __syncthreads();
    a_value = s_schedule_a[tx];
#elif SUBWARP_USE_CP_ASYNC
    // Baseline async path retained for a strict scheduling comparison.
    cp_async_16(&s_b4[tx], b_source);
    cp_async_commit();

    int a_value = input_a[group_id * WARP_SIZE + lane];
#else
    int a_value = input_a[group_id * WARP_SIZE + lane];
#endif

    int sorted_a = warp_bitonic_sort(a_value, lane);

#if SUBWARP_FORCE_EARLY_CP_ASYNC
    // Keep the async wait after the full sort. The shared round-trip gives
    // the wait a real data dependency that ptxas cannot fold.
    s_schedule_a[tx] = sorted_a;
    sorted_a = s_schedule_a[tx];
#endif

#if SUBWARP_USE_CP_ASYNC
    cp_async_wait_all();
    int4 b_values = s_b4[tx];
#else
    int4 b_values = *b_source;
#endif

    unsigned local_valid = 0;

#pragma unroll
    for (int item = 0; item < VALUES_PER_THREAD; ++item) {
        int b = component(b_values, item);
        bool duplicated = warp_binary_search(sorted_a, b);
        local_valid |= static_cast<unsigned>(!duplicated) << item;
    }

    // Pack eight per-thread 4-bit masks into one 32-bit mask per B list.
    unsigned valid_mask = local_valid << (sub_lane * VALUES_PER_THREAD);

#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        valid_mask |=
            __shfl_down_sync(subwarp_mask, valid_mask, offset, SUBWARP_SIZE);
    }

    valid_mask = __shfl_sync(subwarp_mask, valid_mask, 0, SUBWARP_SIZE);
    int count = __popc(valid_mask);

    int compact_base = (warp_id * NUM_B_LISTS + b_id) * COMPACT_STRIDE;

#pragma unroll
    for (int item = 0; item < VALUES_PER_THREAD; ++item) {
        if ((local_valid >> item) & 1u) {
            int position = sub_lane * VALUES_PER_THREAD + item;
            unsigned lower_mask = position == 0 ? 0u : ((1u << position) - 1u);
            int rank = __popc(valid_mask & lower_mask);
            s_compact[compact_base + rank] = component(b_values, item);
        }
    }

    __syncwarp(FULL_MASK);

    int local_sum = 0;

#pragma unroll
    for (int item = 0; item < VALUES_PER_THREAD; ++item) {
        int position = sub_lane * VALUES_PER_THREAD + item;
        if (position < count) {
            local_sum += s_compact[compact_base + position];
        }
    }

#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        local_sum +=
            __shfl_down_sync(subwarp_mask, local_sum, offset, SUBWARP_SIZE);
    }

    int checksum = __shfl_sync(subwarp_mask, local_sum, 0, SUBWARP_SIZE);

    if (sub_lane == 0) {
        int local_result = warp_id * NUM_B_LISTS + b_id;
        s_output_count[local_result] = count;
        s_output_checksum[local_result] = checksum;
    }

    __syncthreads();

    if (tx < RESULTS_PER_BLOCK) {
        size_t output_idx =
            static_cast<size_t>(blockIdx.x) * RESULTS_PER_BLOCK + tx;
        output_count[output_idx] = s_output_count[tx];
        output_checksum[output_idx] = s_output_checksum[tx];
    }
}

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const size_t num_groups = TOTAL_GROUPS;
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
    printf("B layout        : [group][B][element]\n");
    printf("Subwarp mapping : 4 B x 8 threads x 4 values\n");
    printf("A memory        : %.2f MiB\n", a_bytes / 1048576.0);
    printf("B memory        : %.2f MiB\n", b_bytes / 1048576.0);

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
        SUBWARP_KERNEL<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_count,
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
        SUBWARP_KERNEL<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_count,
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
                           "expected_count=16 expected_checksum=%d\n",
                           group, b_id, h_count[idx], h_checksum[idx],
                           expected_checksum);
                }
                ++errors;
            }
        }
    }

    double bytes_per_group = 128.0 + NUM_B_LISTS * 136.0;
    double total_bytes = bytes_per_group * num_groups;
    double seconds = avg_ms / 1000.0;
    double effective_bw = total_bytes / seconds / 1e9;
    double groups_per_sec = num_groups / seconds;
    double pairs_per_sec = num_groups * NUM_B_LISTS / seconds;

    printf("\n%s\n", SUBWARP_VARIANT_LABEL);
    printf("Average time    : %.6f ms\n", avg_ms);
    printf("Groups/s        : %.3f M\n", groups_per_sec / 1e6);
    printf("A-B pairs/s     : %.3f M\n", pairs_per_sec / 1e6);
    printf("Logical traffic : %.3f GB\n", total_bytes / 1e9);
    printf("Effective BW    : %.3f GB/s\n", effective_bw);
    printf("Search stages   : %d / group\n", 15 + 4 * 6);
    printf("Search type     : Generic Binary Search\n");
    printf("Output store    : Block-Coalesced\n");
#if SUBWARP_USE_CP_ASYNC
    printf("B transfer      : cp.async 16B/thread\n");
    printf("Overlap         : %s\n", SUBWARP_OVERLAP_LABEL);
#else
    printf("B transfer      : direct int4 load\n");
#endif
    printf("Wrong results   : %zu\n", errors);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_checksum));
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
