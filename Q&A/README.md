# Warp Candidate Filter 面试问答

这份问答对应简历描述：

> 针对固定 32 元素候选列表，基于 Shared Memory、Warp Register/Shuffle
> 和 Binary Search 重构 Warp-level 1:N 去重内核；结合 Nsight Compute
> 和 SASS 定位 Warp 调度与访存瓶颈，通过 128-bit Vector Load、
> `cp.async` 持续优化数据路径。去重 Kernel 从 4.16 ms 降至 3.02 ms，
> 加速约 1.38x；优化后 DRAM Throughput 约为峰值的 94.9%，主要瓶颈由
> Warp/Search 执行迁移至 DRAM Bandwidth。

## 一分钟项目介绍

这个项目优化的是一个固定规模的 GPU 候选过滤操作。每个 group 有一份
`A[32]`，以及多份 `B[32]`；需要从每份 B 中删除已经存在于 A 的元素，
并将剩余元素紧凑化。默认实验是一个 A 对四个 B，也就是 1:4。

Baseline 让每个 B 元素依次与 A 的 32 个元素做暴力比较，重复执行大量
Warp Shuffle 和整数比较。优化版本先在寄存器中对 A 做一次 32 元素
Bitonic Sort，再让所有 B 元素通过 Warp Shuffle 读取二分搜索的 pivot。
之后又优化了输出合并、B 的线程映射、128-bit 向量加载和 `cp.async`。

在 RTX 4090 上，稳定 benchmark 从约 4.16 ms 降至约 3.02 ms，约
1.38x。这里 4.16 ms 对应 shared-broadcast brute-force baseline。作为瓶颈
对照的 register/shuffle brute-force profile 显示 Compute Throughput 接近
98%、DRAM Throughput 约 56%；优化后 Compute Throughput 降至约 77%，
DRAM Throughput 提升至约 94.9%，说明瓶颈从搜索执行迁移到了显存带宽。

## 核心问题

### 1. 你的这个项目是用来做什么的？

它解决的是 GPU 上固定长度整数列表之间的小规模去重过滤问题。一份 reference
list A 需要被多份 candidate list B 复用；对于每份 B，要判断哪些元素已经
存在于 A，并将未重复元素紧凑化后交给后续计算。

这个操作单次只有 32 个元素，但会在构建过程中执行非常多次，因此
重点不是单次延迟，而是批量执行时的吞吐。项目把这个路径抽成了独立 CUDA
microbenchmark，以便固定输入、输出和 launch 配置后，逐版本分析 Kernel。

### 2. Baseline 是怎么做的，主要瓶颈在哪里？

项目保留了 register/shuffle 和 shared-broadcast 两个暴力基线；简历中的
4.16 ms 使用后者。核心算法都是 Warp-level 暴力搜索：一个 lane 持有一个
B 元素，然后依次读取 A 的 32 个元素并比较。默认一个 group 有四份 B，
因此理论上需要 `4 * 32 * 32 = 4096` 次元素比较。

它的问题不是访存不合并，而是同一份 A 被反复广播，产生大量广播、整数
比较和依赖链。以 register/shuffle v0 的 NCU profile 为例，Compute
Throughput 约 97.9%，而 DRAM Throughput 只有约 56.2%；`No Eligible`
约 72%，说明 Warp 经常因为指令依赖没有可发射的下一条指令。它首先是
Warp/Search 执行瓶颈，而不是纯 DRAM 带宽瓶颈。

### 3. 你是怎么改进的？

主要分成四步：

1. A 只排序一次：每个 lane 将一个 A 元素放在寄存器中，通过 15 个
   `shfl_xor` stage 完成固定 32 元素 Bitonic Sort。
2. 暴力比较改成二分搜索：B 元素通过动态 `shfl` 读取对应 pivot，将每个
   元素的搜索工作从 32 轮降低到固定 5 到 6 轮。
3. 重构数据路径：一个 Warp 划分成四个 8-thread subwarp，每个线程加载
   一个 `int4`，即连续 16 字节；一个 subwarp 正好覆盖一份 128-byte B。
4. 优化结果路径：使用 ballot、popcount 计算 compact rank，在 Shared
   Memory 中消费紧凑结果，再将 count/checksum 以 block-coalesced 方式写回。

另外还实验了二路 ILP、固定 32 专用搜索、`cp.async` 和强制异步调度。
并不是每个版本都更快，最终通过对照实验保留收益稳定的路径。

### 4. 你怎么判断性能已经接近瓶颈？

不是只看 Kernel 时间，而是同时看三类证据：

1. NCU 的 Speed of Light 指标：优化版本 DRAM Throughput 达到约 94.9%，
   实际 Memory Throughput 约 931 GB/s。
2. 瓶颈迁移：Compute Throughput 从基线约 97.9% 降到约 77%，DRAM
   Throughput 从约 56.2% 上升到约 94.9%。
3. 后续实验收益收敛：v1 到 v9 的时间基本集中在 3.02 ms 附近，继续减少
   搜索指令或增加 overlap 没有转化成吞吐提升。

因此准确表述应该是“在当前数据规模、输入布局、输出语义和 RTX 4090 上，
Kernel 已接近 DRAM 带宽上限”，而不是绝对地说再也无法优化。

## 算法与映射

### 5. 这里的 Warp-level 1:N 是什么意思？

1 指一份 `A[32]`，N 指多份 `B[32]`。A 的排序结果可以复用于所有 B，
默认 benchmark 中 `N=4`。它不是一个元素对 N 个元素，而是一份 reference
list 对多份 candidate list。

### 6. 为什么固定为 32 个元素？

32 与一个 CUDA Warp 的 lane 数完全一致。一份 A 可以做到每 lane 一个元素，
排序结果始终保存在 Warp 寄存器中，并通过 Shuffle 交换，不需要构建通用
数据结构。固定长度也允许循环完全展开，减少分支和下标计算开销。

### 7. 一个 Warp 具体怎样处理一组数据？

一个 Warp 处理一个 group。A 的 32 个元素由 32 个 lane 各加载一个并排序。
对于四份 B，Warp 被逻辑划分为四个 8-lane subwarp；每个 lane 持有同一份
B 中连续的四个元素，所以一个 subwarp 处理完整的 `B[32]`。

### 8. 为什么不让一个 Warp 只处理一份 B？

如果每个线程只加载一个 B 元素，单份 B 的读取本身是合并的，但 A 的排序
成本只能服务一份 B。当前 1:4 映射让同一个 Warp 排序一次 A 后处理四份 B，
提高了 A 排序的复用率，同时让 32 个 lane 合计读取连续的 512 字节 B 数据。

### 9. 32 元素 Bitonic Sort 为什么是 15 个 stage？

Bitonic network 对 `2^k` 个元素需要 `k(k+1)/2` 个 compare-exchange stage。
这里 `k=5`，所以是 `5*6/2=15`。每个 stage 使用一次 `shfl_xor` 获取配对
lane 的值，再根据 lane 和 stage 决定保留较大值还是较小值。

### 10. 为什么排序加二分会比直接比较快？

暴力路径对每一份 B 需要 32 轮广播比较。排序的 15 个 stage 只对 A 执行
一次，然后每一波 B 只需要约 6 轮二分。默认四波 B 的 Warp-level 搜索
stage 从约 `4*32=128` 降为 `15+4*6=39`，排序成本被四份 B 分摊。

### 11. 为什么代码中的通用二分是 6 轮，而不是 5 轮？

32 个值的理想决策树深度是 5，但通用实现维护半开区间 `[left,right)`，
并处理等值提前命中和最终边界状态。为保证所有路径在固定展开循环中完成，
实现使用 6 轮。固定 32 专用版本可以压缩控制逻辑，但实测 v3/v4 并没有
超过 v7，因为整体已经逐渐受到 DRAM 限制。

### 12. 二分搜索怎样读取其他 lane 持有的 A？

排序后，第 `i` 个有序 A 元素保存在 lane i 的寄存器中。每个线程独立维护
自己的 `left/right/mid`，然后执行：

```cpp
int value = __shfl_sync(FULL_MASK, sorted_a, mid, 32);
```

`mid` 可以因 lane 而异，Shuffle 会让每个 lane 从自己指定的源 lane 读取
pivot，因此不需要把排序后的 A 写到 Shared Memory。

### 13. 为什么不使用 hash table 或 Bloom filter？

列表只有固定 32 个整数。Hash table 需要初始化、处理冲突和额外 Shared
Memory；Bloom filter 还有误判，不能直接满足精确去重。寄存器排序加二分
具有固定执行结构、没有动态分配，且能直接利用 Warp Shuffle，更适合这个
规模。列表变大或 key 分布改变后，hash 才可能更有优势。

### 14. 为什么不把 B 也排序，然后做 merge？

每个 group 有多份 B，对每份 B 排序会重复支付排序成本，而且 compact 输出
通常还需要保持 B 的原始顺序。当前方案只排序被多次复用的 A，因此成本更低，
输出顺序也能通过 rank 保持。

## Compact 与 Shared Memory

### 15. 去重后的 compact 是怎么做的？

每个线程处理四个 B 元素，并生成四个 valid bit。一个 8-lane subwarp 将
`8*4` 个 bit 合并成 32-bit `valid_mask`。`popc(valid_mask)` 得到总数；
每个有效元素对自己之前的 bit 做 popcount，得到无冲突且保持顺序的 rank，
然后写入 Shared Memory。

### 16. 为什么不能简单使用 `local_pos++`？

`local_pos` 只能知道当前线程内部已经找到几个有效元素，不知道前面其他线程
找到了多少个。多个线程都会从位置 0 开始，导致覆盖。要得到全局紧凑位置，
仍然需要 ballot+popcount、subwarp prefix scan 或 atomic。固定 32 元素时，
mask+popcount 通常比 atomic 更稳定，也天然保持顺序。

### 17. Shared Memory 在这个实现中负责什么？

A 的排序和搜索主要在寄存器与 Shuffle 中完成。Shared Memory 主要用于：

- 保存 compact 后的 B，模拟后续 Kernel 内消费；
- `cp.async` 的 B 落点；
- block-coalesced 输出前的 count/checksum 暂存。

因此不能简单把项目描述成“全部放到 Shared Memory”，核心搜索状态实际在
Warp registers 中。

### 18. NCU 不是仍然报告了 Shared Memory bank conflict 吗？

是的。优化版本在 compact store 上仍有大约 1.5 到 1.6-way conflict，NCU
记录了约 8.39M excessive shared wavefronts。主要原因是四个 8-lane subwarp
同时写四份 B，不同 B 的地址可能映射到相同 bank。

这说明仍有局部优化空间，但不能直接把 NCU 给出的 estimated speedup 当作
端到端收益。当前 DRAM 已接近饱和，额外的地址 swizzle、分阶段写入或 scan
本身也会增加指令，必须以实测判断。

### 19. 为什么 Shared Memory stride 使用 33 而不是 32？

33 让相邻 compact list 的起始 bank 发生偏移，避免所有 list 完全落在相同
bank pattern 上。它能缓解规则性的冲突，但四个 subwarp 同时进行 rank-based
写入时仍不能保证完全无冲突。

### 20. v5 的 block-coalesced output store 做了什么？

原来每个 subwarp 的 leader 直接写全局 count/checksum，Warp 内只有少数 lane
活跃且地址分散。v5 先让 leader 把结果写到 block Shared Memory，block
同步后由连续线程把结果写回全局内存，将稀疏 store 转换成连续 store。

## 向量加载与数据布局

### 21. 128-bit Vector Load 是怎样实现的？

输入保持 `[group][B][element]` 的 B-major 布局。每个 8-lane subwarp 负责
一份 B，每个 lane 通过 `int4` 加载连续四个整数，也就是 16 字节。8 个 lane
合计正好覆盖一份 128-byte B；四个 subwarp 覆盖四份连续 B。

### 22. `int4` 加载为什么不会发生未对齐访问？

`cudaMalloc` 提供足够的基地址对齐；每份 B 是 32 个 int，即 128 字节；
group 和 B 的偏移都是 16 字节的整数倍。因此 `reinterpret_cast<int4*>`
后的每个读取地址都满足 16-byte 对齐要求。

### 23. 这个优化是否偷偷改变了输入布局？

当前 mainline 的 v0 到 v9 都使用相同的 B-major 输入布局，不把 layout
conversion 时间排除在外。早期确实实验过 lane-major/transposed 布局，但已
放入 archive，不作为当前 v6/v7 的性能结论。这一点面试时必须明确。

### 24. 一个 Warp 读取四份 B，会不会产生四次内存事务？

逻辑上每份 B 是连续的 128 字节，整个 Warp 读取四段连续数据。实际 transaction
数量由缓存行、sector 和架构决定，但访问模式是规则且合并的。重点不是保证
“只有一次事务”，而是避免每个线程跨大 stride 读取导致的分散访问。

## cp.async 与 SASS

### 25. 为什么使用 `cp.async`？

目标是把 B 从 Global Memory 直接搬到 Shared Memory，并尝试让搬运与 A 的
寄存器排序重叠。相比先 `LDG` 到寄存器再 `STS`，`cp.async` 可以减少显式
中间寄存器路径，并通过 commit/wait group 表达异步流水。

### 26. 为什么 v7 的 `cp.async` 只快了约 0.05%？

第一，Kernel 已接近 DRAM 带宽上限，单 Warp 的 load latency 也能被大量
resident Warp 隐藏。第二，SASS 显示 `ptxas` 将 v7 的 `LDGSTS` 下沉到了
A 排序末尾，实际 overlap window 很短。因此 `cp.async` 在这里更多是数据
搬运方式变化，而不是形成了很长的计算/访存流水。

### 27. 为什么只看 CUDA 源码不能判断 `cp.async` 是否重叠？

源码顺序不等于机器指令顺序。v7 源码先写 `cp.async`，再加载并排序 A；
但 SASS 中实际是先 `LDG A` 和大部分 Shuffle，之后才出现 `LDGSTS`。
所以必须检查 `cuobjdump/nvdisasm` 生成的 SASS，并定位 `LDGSTS`、`LDGDEPBAR`、
`DEPBAR` 和 `LDS` 的相对位置。

### 28. v8 做了什么，为什么反而更慢？

v8 通过 A 的 volatile Shared Memory round-trip 和 block barrier，强制
`LDGSTS B` 位于完整 A sort 之前，SASS 证明完整 overlap 确实发生了。
但它增加了 A 的 Shared Memory 读写、一次 block barrier 和约 2 KB Shared
Memory。最终约 3.021 ms，比 v7 慢约 0.13%。这说明 overlap 收益小于人为
制造调度依赖的成本。

### 29. v9 为什么要强制 B 在 A 前面？结果怎样？

v9 用两个很小的 noinline device helper 建立真实 CALL 边界：第一个 helper
发射并 commit B，返回后加载和排序 A；第二个 helper 依赖 `sorted_a`，因此
wait 只能出现在排序之后。它不需要把 A 写到 Shared Memory，也不增加新的
block barrier。

SASS 达到了目标，但两个 CALL/RET 有固定成本，五轮均值约 3.030 ms，比 v7
慢约 0.41%。因此“重叠更多”不等于“整体更快”。

### 30. 这些失败版本为什么还保留？

它们证明了几个容易被误判的问题：源码顺序不代表 SASS 顺序；完整 overlap
不一定有收益；为了强迫调度而增加的 barrier、Shared Memory 或 CALL 可能
比被隐藏的 latency 更贵。保留可复现的负向实验，比只展示最快版本更能说明
优化过程是基于证据而不是碰参数。

## Profiling 与性能判断

### 31. 你主要看了哪些 Nsight Compute 指标？

主要包括：

- `Duration`：Kernel 时间；
- `DRAM Throughput`：是否接近显存峰值；
- `Compute (SM) Throughput`：执行管线压力；
- `No Eligible` 和 `Eligible Warps Per Scheduler`：Warp 是否经常无法发射；
- Warp stall reason：定位 scoreboard、memory、barrier 等等待；
- registers、Shared Memory、occupancy：判断资源限制；
- shared wavefront excessive：定位 bank conflict；
- executed instructions：判断优化是否以额外指令换取了收益。

### 32. 你说的 934 GB/s 和 94.9% 是同一个指标吗？

不是。程序打印的约 934 GB/s 是按预先定义的 logical traffic 除以 Kernel
时间计算的“逻辑有效带宽”，适合版本间对比。NCU 的约 930.7 GB/s 和
94.75% 才是硬件计数器给出的实际 Memory/DRAM Throughput。

面试时应分别说明，不能把程序推算值冒充硬件实测值。

### 33. 为什么 NCU 中的 Kernel 时间可能和 benchmark 不完全一致？

NCU 为采集计数器会 replay Kernel、控制时钟或引入 profiling 环境，时间不
一定等于正常运行。最终版本时间来自 CUDA Event、warmup 后重复执行的
benchmark；瓶颈归因来自 NCU。二者承担不同作用，不应混在同一组数字里。

### 34. 1.38x 是怎样计算的？

以约 4.16 ms 的 Shared-broadcast brute-force baseline 和约 3.02 ms 的
优化版本为例：

```text
speedup = 4.16 / 3.02 ≈ 1.38x
time reduction = (4.16 - 3.02) / 4.16 ≈ 27.4%
```

“加速 1.38x”和“耗时降低约 27%”是同一结果的两种表达，不应说成性能
提升 38% 且耗时降低 38%。

### 35. 为什么 v1 到 v9 的时间几乎一样？

v1 已经通过排序加二分大幅减少搜索执行，Kernel 随后迅速接近 DRAM 上限。
后续版本优化的是更小的局部路径，例如 ILP、固定轮数、store 合并和 B load。
这些优化可能改善某项微观指标，但在带宽瓶颈下只能带来千分级变化，甚至被
新增指令抵消。

### 36. Occupancy 是不是越高越好？

不是。v7 的 achieved occupancy 约 96.4%，27 registers/thread，已经有足够
Warp 隐藏延迟。继续降低寄存器、追求 100% occupancy 不一定增加带宽，反而
可能引入 spill 或增加指令。Occupancy 是约束条件，不是最终优化目标。

### 37. `No Eligible` 仍有约 44%，为什么还说是带宽瓶颈？

`No Eligible` 表示某些周期 scheduler 没有可发射 Warp，原因可能是内存或
scoreboard 依赖。优化版本同时达到约 95% DRAM Throughput，说明这些等待
主要发生在饱和的内存数据路径上。不能只看一个 stall 指标，需要与 DRAM、
Compute Throughput 和 SASS 一起判断。

## 实验设计与严谨性

### 38. 你怎样保证各版本比较公平？

所有 mainline 版本使用相同的数据布局、group 数量、输出语义、block size
和编译参数；输入初始化不计入 Kernel 时间。每个程序先 warmup，再用 CUDA
Event 对多次 launch 计时，并执行多轮外层 benchmark。每个版本都必须通过
host correctness validation。

### 39. 正确性是怎样验证的？

测试数据被确定性构造为每份 B 中 16 个元素存在于 A、16 个元素不存在。
Kernel 输出 survivor count 和 compact 后数据的 checksum，host 端检查所有
group 的 count 与 checksum，要求 `Wrong results: 0`。

checksum 不能替代生产环境的完整逐元素验证，但能防止编译器删除 compact
读写路径，并为 microbenchmark 提供低成本回归检查。

### 40. 为什么输出 checksum，而不是完整候选数组？

这个仓库重点隔离 Kernel 内过滤、compact 和后续消费路径。将 compact 数据
重新全部写回 Global Memory 会引入另一条大流量输出路径，掩盖目标操作。
checksum 让 Shared compact 结果必须被真实读取，同时只写固定大小的验证
结果。生产集成时应另外验证真实输出接口，不能直接把 checksum benchmark
等同于完整系统实现。

### 41. 测试数据会不会过于理想化？

会，这是 microbenchmark 的限制。当前数据固定为 50% duplicate，且各 group
工作量一致，有利于稳定比较版本，但不能覆盖真实数据中的重复率、分布偏斜、
缓存复用和尾部长度。严谨结论应限定在固定 32、1:4 和当前分布下，并补充
不同 duplicate ratio 与数据集的敏感性实验。

### 42. 为什么 RTX 3090 上版本差距可能不同？

3090 与 4090 的显存带宽、缓存、SM 数量、时钟、调度器和 `cp.async` 实现
不同。一个在 4090 上已到 DRAM ceiling 的版本，在 3090 上可能更早受带宽
限制；也可能因为指令吞吐和 latency hiding 不同，使 ILP 或异步加载的收益
改变。因此不能直接移植绝对时间，应在目标 GPU 上重新编译、检查 SASS 和
采集 NCU。

### 43. 如何避免 GPU 温度和动态频率影响结论？

运行前 warmup，保持 power limit、application clock、温度和后台负载一致；
进行多轮交错测试，而不是先连续跑完一个版本；报告均值和波动范围。千分级
差异如果没有多轮统计和 SASS/NCU 证据，不应宣称为稳定优化。

## 设计取舍与追问

### 44. 二路 ILP 为什么没有明显加速？

二路 ILP 尝试同时推进两个 B 的二分搜索，以隐藏 Shuffle 和整数依赖。但它
增加了搜索状态和寄存器压力，调度器本身已经有大量 resident Warp；当 DRAM
成为主瓶颈后，增加线程内 ILP 很难提升整体吞吐。v2 与 v1 的时间基本持平。

### 45. 为什么固定 32 专用搜索也没有超过通用版本？

专用搜索能减少部分循环和边界判断，但它不会减少必须读取的 A/B 数据，也
不会提高显存峰值。节省的少量整数指令可能被更复杂的控制或寄存器调度抵消。
因此 v3/v4 的时间与 v1/v2 接近。

### 46. 下一步还可以怎样优化？

优先级较高的实验是 compact Shared Memory 的 bank-aware swizzle，因为 NCU
已经定位到明确的 excessive wavefront。还可以测试不同 duplicate ratio、
1:1/1:2/1:4/1:8、block size，以及在真实调用方中将过滤结果直接留在设备端，
减少 Kernel 边界和 CPU-GPU 同步。

但预期收益必须保守：当前 DRAM Throughput 已约 95%，单 Kernel 内继续优化
大概率只有小幅收益。更大的系统收益可能来自减少全局数据流量或 Kernel
launch，而不是继续压缩二分的几条指令。

### 47. 如果 N 不是 4，这个映射还能用吗？

算法可以扩展，但当前 4 个 8-lane subwarp 的映射是针对 N=4 专门设计的。
N=1 或 N=2 时会有 lane 利用率或映射策略问题；N 更大时需要分批处理。A
排序成本的摊销也会随 N 改变，所以不同 N 应单独选择 mapping，而不是直接
假设 v7 始终最优。

### 48. 如果候选长度不是 32 怎么办？

小于 32 可以使用 active mask 和 padding，但会引入边界处理；大于 32 则
不能再用一个 Warp 的每 lane 一个 A 元素，需要多 Warp、分块或其他数据
结构。这个项目的性能来自固定 32 的专用化，因此不能无成本推广到任意长度。

### 49. 你个人在这个项目中的核心贡献是什么？

可以概括为三点：

1. 将 1:N 去重抽象成可复现、可验证的 CUDA microbenchmark，并建立公平的
   版本对照；
2. 设计 A 寄存器排序、Warp Shuffle 二分、4x8 subwarp/int4 数据映射和
   block-coalesced 输出路径；
3. 使用 NCU 与 SASS 验证瓶颈迁移和真实指令顺序，并保留 v8/v9 等负向实验，
   避免把源码层面的“看起来异步”误判成机器层面的有效 overlap。

回答时应根据自己的实际工作范围调整，不要把未参与的系统集成或端到端收益
归到自己名下。

### 50. 这个项目最大的技术难点是什么？

最大的难点不是写出一个能工作的 Kernel，而是区分三件事：源码表达的优化、
SASS 实际执行的优化、最终吞吐真正受益的优化。例如 v7 源码先发射
`cp.async`，但 SASS 将其下沉；v8/v9 成功强制完整 overlap，却因为 barrier
或 CALL 成本更慢。只有把算法、编译器调度和硬件计数器串起来，才能得到可信
结论。

## 容易被追问的口径

- 不要说“v7 的 `cp.async` 完整隐藏了 B load”。SASS 证明其 overlap 很短。
- 不要把程序计算的 logical effective BW 当成 NCU 的真实 DRAM bandwidth。
- 不要说“达到 94.9% 后绝对无法再优化”，应限定 GPU、数据和输出语义。
- 不要说所有版本都改变了输入布局；当前 mainline 使用相同 B-major 布局。
- 不要把 Kernel microbenchmark 的 1.38x 直接说成完整应用的 E2E 1.38x。
- 不要只报最快时间，应说明 warmup、重复次数、正确性和 profiling 方法。
