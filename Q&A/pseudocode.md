
__device__ __forceinline__ void cp_async_16(void* shared_ptr, const void* global_ptr)
{
    unsigned shared_address = static_cast<unsigned>(__cvta_generic_to_shared(shared_ptr));
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




### <<<nodes, 16 * 32>>> 16groups 且 1个warp负责一组(A, 4B)
__global__ void kernel(){
    
    首先是 shared memory， 都是 16 份的。
    s_compact[16][4][33]    // 16份 的 4 组 B

    s_output_count[64]      // 16份 的 4组 B

    s_output_checksum[64]   // 16份 的 4组 B

    __shared__ __align__(16) int4 s_b4[BLOCK_SIZE];  // 每个thread就是负责一个 int4
    因为这里的 s_b4 是给 cp.async 做 16 字节 128bit 搬运 用的，所以要保证目标地址满足 16B 对齐。
    

    计算组B的数据地址对应的/ 向量化访存就是 目前这里1个thread负责4个int32.
    const int4* input_b4 = reinterpret_cast<const int4*>(input_b);

    const int4* b_source = input_b4 + group_id * WARP_SIZE + lane;

    每个 thread 发起一次 cp.async,把自己负责的 16B B 数据：从global memory到shared memory
    cp_async_16(&s_b4[tx], b_source);
    cp_async_commit();

    开始读取组A的元素
    a_value = input_a[group_id * WARP_SIZE + lane];

    双调排序，每个warp都是这样执行的。
    sorted_a = warp_bitonic_sort(a_value, lane);

    cp_async_wait_all();    // 等B的数据全部搬运好。


    开始二阶段：
    int4 b_values = s_b4[tx];

    unsigned local_valid = 0;
    #pragma unroll
    for(int i = 0; i < 4; i++){
        if(i == 0)int b = b_values.x;
        if(i == 1)int b = b_values.y;
        if(i == 2)int b = b_values.z;
        if(i == 3)int b = b_values.w;
        
        bool dp = warp_binary_search(sorted_a, b);
        if (!dp) local_valid |= (1u << i); 
        没重复就把 local_valid 对应bit置1
    }


    把同一个 8-thread subwarp 里 8 个线程各自的 4-bit local_valid 拼成一个完整的 32-bit valid_mask
    valid_mask = local_valid << (sub_lane * 4);
}



__device__ __forceinline__ bool warp_binary_search(int sorted_a, int target)
{
    int left = 0;
    int right = WARP_SIZE;
    bool existed = false;

    #pragma unroll
    for (int step = 0; step < 6; step++) {
        bool active = !existed && left < right;

        int mid = active ? ((left + right) >> 1) : 0;
        int value = __shfl_sync(FULL_MASK,sorted_a,mid,WARP_SIZE);

        if (active) {
            if (value == target) existed = true;
            else if (value < target) left = mid + 1;
            else right = mid;
        }
    }
    return existed;
}