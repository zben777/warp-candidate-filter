(base) bzhang@galaxy:~/code/2026/cagra_kernel$ sudo $(which ncu) \
  --kernel-name regex:brute_force_shared_kernel \
  --launch-skip 10 \
  --launch-count 1 \
  --set full \
  -o v0_shared_4b_ncu \
  ./v0_shared_4b
[sudo] password for bzhang: 
==PROF== Connected to process 269728 (/home/bzhang/code/2026/cagra_kernel/v0_shared_4b)
GPU             : NVIDIA GeForce RTX 4090
Grid            : <<<262144, 512>>>
Warps/block     : 16
Groups          : 4194304
NUM_B_LISTS     : 4
A memory        : 512.00 MiB
B memory        : 2048.00 MiB
==PROF== Profiling "brute_force_shared_kernel": 0%....50%....100% - 35 passes

V0-Shared 1A-NB Shared Broadcast Brute Force
Average time    : 61.991608 ms
Groups/s        : 67.659 M
A-B pairs/s     : 270.637 M
Logical traffic : 2.819 GB
Effective BW    : 45.467 GB/s
Search rounds   : 128 / group
Wrong results   : 0
==PROF== Disconnected from process 269728
==PROF== Report: /home/bzhang/code/2026/cagra_kernel/v0_shared_4b_ncu.ncu-rep
(base) bzhang@galaxy:~/code/2026/cagra_kernel$ sudo $(which ncu) \
  --import v0_shared_4b_ncu.ncu-rep \
  --page details
[269728] v0_shared_4b@127.0.0.1
  brute_force_shared_kernel(const int *, const int *, int *, int *) (262144, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- -------------
    Metric Name               Metric Unit  Metric Value
    ----------------------- ------------- -------------
    DRAM Frequency          cycle/nsecond         10.24
    SM Frequency            cycle/nsecond          2.23
    Elapsed Cycles                  cycle    11,243,212
    Memory Throughput                   %         99.68
    DRAM Throughput                     %         56.91
    Duration                      msecond          5.03
    L1/TEX Cache Throughput             %         99.73
    L2 Cache Throughput                 %         23.90
    SM Active Cycles                cycle 11,236,827.84
    Compute (SM) Throughput             %         99.68
    ----------------------- ------------- -------------

    INF   The kernel is utilizing greater than 80.0% of the available compute or memory performance of the device. To   
          further improve performance, work will likely need to be shifted from the most utilized to another unit.      
          Start by analyzing workloads in the Compute Workload Analysis section.                                        

    Section: GPU Speed Of Light Roofline Chart
    INF   The ratio of peak float (fp32) to double (fp64) performance on this device is 64:1. The kernel achieved 0% of 
          this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel Profiling Guide       
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline      
          analysis.                                                                                                     

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         1.14
    Executed Ipc Elapsed  inst/cycle         1.14
    Issue Slots Busy               %        28.58
    Issued Ipc Active     inst/cycle         1.14
    SM Busy                        %        33.44
    -------------------- ----------- ------------

    INF   ALU is the highest-utilized pipeline (23.9%) based on active cycles, taking into account the rates of its     
          different instructions. It executes integer and logic operations. It is well-utilized, but should not be a    
          bottleneck.                                                                                                   

    Section: Memory Workload Analysis
    --------------------------- ------------ ------------
    Metric Name                  Metric Unit Metric Value
    --------------------------- ------------ ------------
    Memory Throughput           Gbyte/second       559.57
    Mem Busy                               %        49.92
    Max Bandwidth                          %        99.68
    L1/TEX Hit Rate                        %        25.69
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Hit Rate                            %        28.58
    Mem Pipes Busy                         %        99.68
    --------------------------- ------------ ------------

    Section: Memory Workload Analysis Tables
    OPT   Estimated Speedup: 6.875%                                                                                     
          The memory access pattern for stores from L1TEX to L2 is not optimal. The granularity of an L1TEX request to  
          L2 is a 128 byte cache line. That is 4 consecutive 32-byte sectors per L2 request. However, this kernel only  
          accesses an average of 1.0 sectors out of the possible 4 sectors per cache line. Check the Source Counters    
          section for uncoalesced stores and try to minimize how many cache lines need to be accessed per memory        
          request.                                                                                                      

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        28.58
    Issued Warp Per Scheduler                        0.29
    No Eligible                            %        71.42
    Active Warps Per Scheduler          warp        10.77
    Eligible Warps Per Scheduler        warp         1.42
    ---------------------------- ----------- ------------

    OPT   Estimated Speedup: 71.42%                                                                                     
          Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only      
          issues an instruction every 3.5 cycles. This might leave hardware resources underutilized and may lead to     
          less optimal performance. Out of the maximum of 12 warps per scheduler, this kernel allocates an average of   
          10.77 active warps per scheduler, but only an average of 1.42 warps were eligible per cycle. Eligible warps   
          are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible   
          warp results in no instruction being issued and the issue slot remains unused. To increase the number of      
          eligible warps, avoid possible load imbalances due to highly different execution durations per warp.          
          Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.            

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        37.70
    Warp Cycles Per Executed Instruction           cycle        37.70
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    30.43
    ---------------------------------------- ----------- ------------

    OPT   Estimated Speedup: 63.64%                                                                                     
          On average, each warp of this kernel spends 24.0 cycles being stalled waiting for the MIO (memory             
          input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of  
          the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory        
          instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline  
          pressure. This stall type represents about 63.6% of the total average of 37.7 cycles between issuing two      
          instructions.                                                                                                 
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on         
          sampling data. The Kernel Profiling Guide                                                                     
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details    
          on each stall reason.                                                                                         

    Section: Instruction Statistics
    ---------------------------------------- ----------- -------------
    Metric Name                              Metric Unit  Metric Value
    ---------------------------------------- ----------- -------------
    Avg. Executed Instructions Per Scheduler        inst     3,211,264
    Executed Instructions                           inst 1,644,167,168
    Avg. Issued Instructions Per Scheduler          inst  3,211,321.51
    Issued Instructions                             inst 1,644,196,612
    ---------------------------------------- ----------- -------------

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   512
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                262,144
    Registers Per Thread             register/thread              34
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block            4.10
    Threads                                   thread     134,217,728
    Waves Per SM                                              682.67
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block            3
    Block Limit Shared Mem                block            6
    Block Limit Warps                     block            3
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        89.66
    Achieved Active Warps Per SM           warp        43.04
    ------------------------------- ----------- ------------

    OPT   Estimated Speedup: 10.34%                                                                                     
          This kernel's theoretical occupancy is not impacted by any block limit. The difference between calculated     
          theoretical (100.0%) and measured achieved occupancy (89.7%) can be the result of warp scheduling overheads   
          or workload imbalances during the kernel execution. Load imbalances can occur between warps within a block    
          as well as across blocks of the same kernel. See the CUDA Best Practices Guide                                
          (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on           
          optimizing occupancy.                                                                                         

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.00
    Branch Instructions              inst    4,194,304
    Branch Efficiency                   %            0
    Avg. Divergent Branches                          0
    ------------------------- ----------- ------------

(base) bzhang@galaxy:~/code/2026/cagra_kernel$ 