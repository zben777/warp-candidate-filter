# Warp Candidate Filter 面试问答

这份问答对应简历描述：

> 针对固定 32 元素候选列表，基于 Shared Memory、Warp Register/Shuffle
> 和 Binary Search 重构 Warp-level 1:N 去重内核；结合 Nsight Compute
> 和 SASS 定位 Warp 调度与访存瓶颈，通过 128-bit Vector Load、
> `cp.async` 持续优化数据路径。去重 Kernel 从 4.16 ms 降至 3.02 ms，
> 加速约 1.38x；优化后 DRAM Throughput 约为峰值的 94.9%，主要瓶颈由
> Warp/Search 执行迁移至 DRAM Bandwidth。

## 回答口径与证据边界

面试时先区分三层结论，避免把不同实验的数字混在一起：

| 层次 | 可以证明什么 | 主要证据 |
|---|---|---|
| 独立 Kernel benchmark | 固定 `A[32]`、四份 `B[32]` 下各版本的时间、吞吐和正确性 | `src/`、`scripts/benchmark.sh`、`profiles/*/*_result.txt` |
| Kernel 瓶颈分析 | Warp/Search、DRAM、bank conflict、occupancy 和真实指令顺序 | `profiles/*/*_ncu_details.txt`、`.ncu-rep`、`.sass` |
| 上层系统收益 | 减少同步、设备端复用、简历中的 1.76x-2.26x E2E | 内部集成代码、系统日志和数据集实验，不由这个公开 microbenchmark 单独证明 |

简历中的 `4.16 ms -> 3.02 ms` 使用 shared-broadcast brute-force 和推荐版本的
五轮均值并做两位小数取整，`4.16 / 3.02 = 1.38x`。仓库中某一次重新采集的
profile snapshot 可能是约 `4.46 ms`，而 README 的稳定五轮参考值是
`4.154 ms`。前者用于读取硬件计数器，后者用于报告正常运行时间；不能从两个
不同采集批次各挑一个数字计算 speedup。

`94.9%` 是 Nsight Compute 的硬件计数器比例；程序打印的约 `934 GB/s` 是按
固定 logical traffic 推算的有效带宽。两者数值接近，但含义不同。

## 项目介绍

### 30 秒版本

我优化的是一个固定长度的 Warp-level 1:N 候选过滤 Kernel：一份 `A[32]`
需要过滤四份 `B[32]`。Baseline 对每个 B 元素暴力扫描 32 个 A，存在长的
Shuffle/compare 依赖链。我把它改成 A 在 Warp 寄存器中只做一次 Bitonic
Sort，B 再通过 Shuffle 读取 pivot 做二分，并继续优化 compact、4x8 subwarp
映射、`int4` 128-bit load 和 `cp.async`。RTX 4090 上时间从约 4.16 ms 降到
3.02 ms，1.38x；NCU 显示瓶颈从搜索执行迁移到约 94.9% DRAM Throughput。

### 一分钟版本

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

最后我没有只保留最快版本。v8/v9 分别用 Shared Memory 调度边界和 noinline
调用强制把异步 B 搬运放到 A 排序之前，SASS 证明 overlap 确实扩大了，但
barrier、Shared Memory round-trip 或 CALL/RET 的代价超过收益。这说明我的
优化过程不是只看 CUDA 源码，而是用 benchmark、NCU 和 SASS 闭环验证。

## 核心问题

### 1. 你的这个项目是用来做什么的？

它解决的是 GPU 上固定长度整数列表之间的小规模去重过滤问题。一份 reference
list A 需要被多份 candidate list B 复用；对于每份 B，要判断哪些元素已经
存在于 A，并将未重复元素紧凑化后交给后续计算。

这个操作单次只有 32 个元素，但会在构建过程中执行非常多次，因此
重点不是单次延迟，而是批量执行时的吞吐。项目把这个路径抽成了独立 CUDA
microbenchmark，以便固定输入、输出和 launch 配置后，逐版本分析 Kernel。

**回答时要强调的边界：** 公开项目输出的是 survivor count 和 checksum，
目的是隔离过滤、compact 和设备端消费路径，并不等价于完整业务系统。若面试官
追问系统价值，可以回答：这类小列表操作单次工作量不大，但调用次数非常多；
把过滤留在设备端可避免中间结果往返和频繁同步。独立仓库证明的是 Kernel
机制和性能结论，上层 E2E 收益要由另外的系统实验支撑。

**可能追问：为什么值得单独优化这么小的列表？**

因为总体成本是“单次成本乘以调用规模”。默认一次 launch 处理 4,194,304
个 group，即 16,777,216 个 A-B pair。固定 32 使单次工作适合 Warp 专用化，
而大批量并行又足以占满 GPU，所以它是典型的 throughput microkernel。

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

更具体地说，register/shuffle baseline 中一轮会把某个 lane 的 A 广播给整个
Warp，所有 lane 比较后再进入下一轮。四份 B 共 `4 * 32 = 128` 个 Warp-level
搜索 round，元素级比较数是 `4 * 32 * 32 = 4096`。这些 round 之间存在数据
和控制依赖，调度器不能像处理大量互不依赖算术那样自由穿插，因此单看“比较
只是整数指令”会低估它的调度成本。

shared-broadcast baseline 把 A 放入 Shared Memory，由 Warp 每轮读取同一个
地址，利用 broadcast 语义避免同地址 bank conflict；但它仍然没有消除 32 轮
串行搜索。Shared Memory 只改变 A 的供数路径，没有改变算法复杂度。

**可能追问：为什么 `No Eligible` 高却不是内存瓶颈？**

`No Eligible` 只表示 scheduler 某些周期没有可发射 Warp，不直接给出根因。
Baseline 同时呈现 Compute Throughput 约 97.9%、DRAM 约 56.2%，并且 SASS
中有密集的 Shuffle/比较依赖链，所以应归因于执行和依赖，而不是 DRAM 饱和。

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

这四步分别解决不同层次的问题：

| 层次 | Baseline 问题 | 改动 | 预期效果 |
|---|---|---|---|
| 算法 | 每个 B 扫描全部 A | A 排序一次，B 二分 | 搜索 stage 从约 128 降至 39 |
| Warp 映射 | 每 lane 的工作与连续加载不匹配 | 4 个 8-lane subwarp，每 lane 处理 4 个相邻值 | 保持 1:4 并形成 16-byte 连续线程加载 |
| Compact | 有效元素需要跨线程排位置 | ballot/bit mask + popcount rank | 无 atomic、保持 B 原顺序 |
| 输出 | 少数 leader 分散写 global | block Shared Memory 暂存后连续线程写回 | 改善 global store 合并性 |
| 搬运 | Global -> register -> Shared | 16-byte `cp.async` | 尝试降低中间寄存器路径并与排序重叠 |

最主要的收益来自第一步，即算法复杂度和 Warp stage 数下降。v5-v9 属于到达
带宽上限后的数据路径实验，收益明显更小。面试时不能把 1.38x 全部归因于
`int4` 或 `cp.async`；从 v6 到 v7 的稳定收益只有约 0.05%。

**可能追问：你如何确定每一步有效？**

每个想法都有独立版本，不在一个版本里同时改变多个变量；用同一输入契约跑
正确性和多轮 benchmark，再用 NCU 看指标是否按假设变化，用 SASS 检查
编译器是否真的生成了目标指令和顺序。

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

瓶颈迁移的原因可以用“单位字节计算量下降”解释：A、B 和结果的 logical
traffic 基本没有变，但搜索 stage 从 128 降到约 39，单位输入字节对应的
Shuffle/compare 明显减少。于是执行管线先被释放，内存供数速度开始决定总
时间。在 roofline 语言里，就是 arithmetic/operation intensity 降低后，工作点
从执行上限移动到带宽屋顶。

这里有四个相互印证的观察：

- 时间：约 `4.16 -> 3.02 ms`；
- 执行：Compute Throughput 从接近 98% 降到约 77%；
- 内存：DRAM Throughput 从约 56% 升到约 94.9%；
- 边际收益：v1 之后的版本大多在 3.02-3.04 ms，强制更多 overlap 也不快。

**可能追问：95% 为什么不是 100%？**

峰值或 peak sustained 是特定测试下的上限，真实 Kernel 还包含指令、地址生成、
同步、Shared Memory 和读写混合，通常无法长期达到理论 100%。95% 说明继续
优化已有字节的搬运空间很小，但如果能减少总字节数、跨 Kernel 融合或改变
输出契约，仍可能继续降低时间。

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

以默认 `BLOCK_SIZE=512` 为例，v7 的静态 Shared Memory 可以精确拆成：

```text
s_compact         = 16 warps * 4 B * 33 ints = 8,448 bytes
s_output_count    = 64 ints                  =   256 bytes
s_output_checksum = 64 ints                  =   256 bytes
s_b4              = 512 int4                 = 8,192 bytes
total                                         17,152 bytes
```

v8 再增加 `s_schedule_a[512]`，所以是 `17,152 + 2,048 = 19,200 bytes`。
这些数字既能从源码计算，也与 NCU 的 17.15/19.20 KB 报告对应。

### 18. NCU 不是仍然报告了 Shared Memory bank conflict 吗？

是的。优化版本在 compact store 上仍有大约 1.5 到 1.6-way conflict，NCU
记录了约 8.39M excessive shared wavefronts。主要原因是四个 8-lane subwarp
同时写四份 B，不同 B 的地址可能映射到相同 bank。

这说明仍有局部优化空间，但不能直接把 NCU 给出的 estimated speedup 当作
端到端收益。当前 DRAM 已接近饱和，额外的地址 swizzle、分阶段写入或 scan
本身也会增加指令，必须以实测判断。

**如何定位是哪条指令冲突？** 先在 NCU Source 页面将 excessive wavefront
关联到源码/SASS，再检查同一条 `STS` 在一个 warp instruction 中各 lane 的
地址。bank conflict 是“同一条 Shared Memory 指令内”的地址映射问题，不是
看到多个线程最终写了同一个 shared 数组就能直接下结论。

### 19. 为什么 Shared Memory stride 使用 33 而不是 32？

33 让相邻 compact list 的起始 bank 发生偏移，避免所有 list 完全落在相同
bank pattern 上。它能缓解规则性的冲突，但四个 subwarp 同时进行 rank-based
写入时仍不能保证完全无冲突。

对 32-bit 数据，可以近似用 `bank = word_address % 32` 分析。stride 32 会让
每份 list 的 base 落在同一个 bank；stride 33 让第 `k` 份 list 的 base 偏移
`k` 个 bank。不过 rank 由运行时 valid mask 决定，四个 subwarp 的动态 rank
仍可能重合到相同 bank，因此 padding 只能缓解，不能证明 conflict-free。

### 20. v5 的 block-coalesced output store 做了什么？

原来每个 subwarp 的 leader 直接写全局 count/checksum，Warp 内只有少数 lane
活跃且地址分散。v5 先让 leader 把结果写到 block Shared Memory，block
同步后由连续线程把结果写回全局内存，将稀疏 store 转换成连续 store。

## 向量加载与数据布局

### 21. 128-bit Vector Load 是怎样实现的？

输入保持 `[group][B][element]` 的 B-major 布局。每个 8-lane subwarp 负责
一份 B，每个 lane 通过 `int4` 加载连续四个整数，也就是 16 字节。8 个 lane
合计正好覆盖一份 128-byte B；四个 subwarp 覆盖四份连续 B。

选择 128-bit 有三层原因：

1. `int4` 是 CUDA 中自然的 16-byte 对齐向量类型，一条线程级加载表达四个
   相邻 32-bit 元素，减少独立 load 指令和地址计算；
2. 一份 `B[32]` 恰好是 `32 * 4 = 128 bytes`，由 8 个线程各取 16 bytes，
   映射正好完整覆盖且没有重叠；
3. 四个 8-lane subwarp 同时覆盖四份 B，Warp 32 个 lane 全部参与加载，
   不会为了向量化牺牲 SIMD lane 利用率。

“128-bit load”指每个活跃线程请求 16 bytes，并不表示整个 Warp 只产生一个
128-bit 内存事务。Warp 级事务仍会按地址分布拆成 cache line/sector 请求。
最终要通过 SASS 中的宽加载或 `LDGSTS ... 16B` 以及 NCU transaction 指标确认，
不能只因源码写了 `int4` 就宣称硬件一定按理想方式执行。

**可能追问：为什么不是 `int2` 或每线程 8 个元素？**

`int2` 会让每份 B 需要 16 个线程，1:4 时需要 64 lanes，无法由一个 Warp
同时覆盖。每线程 8 个元素需要 32-byte 线程级搬运，通常拆成多条指令，也会
增加线程私有状态。每线程 4 个元素正好平衡了 1:4 映射、16-byte 原生搬运和
寄存器压力。

### 22. `int4` 加载为什么不会发生未对齐访问？

`cudaMalloc` 提供足够的基地址对齐；每份 B 是 32 个 int，即 128 字节；
group 和 B 的偏移都是 16 字节的整数倍。因此 `reinterpret_cast<int4*>`
后的每个读取地址都满足 16-byte 对齐要求。

若真实调用方传入带偏移的子指针，就不能只依赖 `cudaMalloc` 的原始基址。
接口必须保证 `reinterpret_cast<uintptr_t>(ptr) % 16 == 0`，并保证 stride
也是 16 的倍数；否则应走标量 fallback，或用安全的非对齐搬运实现。把未对齐
地址直接解释为 `int4*` 既可能降低性能，也可能违反 C++ 对齐要求，不能接受。

### 23. 这个优化是否偷偷改变了输入布局？

当前 mainline 的 v0 到 v9 都使用相同的 B-major 输入布局，不把 layout
conversion 时间排除在外。早期确实实验过 lane-major/transposed 布局，但已
放入 archive，不作为当前 v6/v7 的性能结论。这一点面试时必须明确。

### 24. 一个 Warp 读取四份 B，会不会产生四次内存事务？

逻辑上每份 B 是连续的 128 字节，整个 Warp 读取四段连续数据。实际 transaction
数量由缓存行、sector 和架构决定，但访问模式是规则且合并的。重点不是保证
“只有一次事务”，而是避免每个线程跨大 stride 读取导致的分散访问。

在 v6 映射中 lane 0-7 读取 B0 的 8 个连续 `int4`，lane 8-15 读取 B1，依次
类推。四份 B 在 `[group][B][element]` 中彼此连续，所以整个 Warp 覆盖连续
512 bytes。硬件会把它拆成多个 sector/request，但每个 sector 的有效字节率高。

## cp.async 与 SASS

### 25. 为什么使用 `cp.async`？

目标是把 B 从 Global Memory 直接搬到 Shared Memory，并尝试让搬运与 A 的
寄存器排序重叠。相比先 `LDG` 到寄存器再 `STS`，`cp.async` 可以减少显式
中间寄存器路径，并通过 commit/wait group 表达异步流水。

代码中每个线程执行一条 16-byte：

```cpp
cp.async.cg.shared.global [shared_addr], [global_addr], 16;
cp.async.commit_group;
// independent A work
cp.async.wait_group 0;
```

`commit_group` 是提交当前异步 copy group，不是等待完成；`wait_group 0` 才要求
之前提交的 group 全部完成。等待之后才能安全读取 `s_b4`。`.cg` 表示 global
访问采用 cache-global 策略，重点是 Global-to-Shared 的异步数据路径，而不是
“完全绕开所有 cache”。

`cp.async` 优化的是 latency hiding 和中间搬运路径，不会减少 B 的 DRAM
字节数。因此当 Kernel 已经受 sustained bandwidth 限制时，它通常只能带来
很小收益；只有存在足够独立计算窗口且原路径受 load latency/寄存器搬运限制，
收益才可能明显。

### 26. 为什么 v7 的 `cp.async` 只快了约 0.05%？

第一，Kernel 已接近 DRAM 带宽上限，单 Warp 的 load latency 也能被大量
resident Warp 隐藏。第二，SASS 显示 `ptxas` 将 v7 的 `LDGSTS` 下沉到了
A 排序末尾，实际 overlap window 很短。因此 `cp.async` 在这里更多是数据
搬运方式变化，而不是形成了很长的计算/访存流水。

这里不存在“B 阻塞 A”的源码级必然关系。异步 copy 发射后，A 的普通 Global
load 和 Shuffle sort 可以继续执行；真正的问题是编译器有权重排无依赖指令，
最后生成的 SASS 没有保留源码中设想的长 overlap。是否阻塞要看 scoreboard、
依赖关系和机器指令位置，而不是看两行 CUDA 的先后。

### 27. 为什么只看 CUDA 源码不能判断 `cp.async` 是否重叠？

源码顺序不等于机器指令顺序。v7 源码先写 `cp.async`，再加载并排序 A；
但 SASS 中实际是先 `LDG A` 和大部分 Shuffle，之后才出现 `LDGSTS`。
所以必须检查 `cuobjdump/nvdisasm` 生成的 SASS，并定位 `LDGSTS`、`LDGDEPBAR`、
`DEPBAR` 和 `LDS` 的相对位置。

本项目观察到的关键顺序是：

```text
v7: LDG A -> 大部分 A sort -> LDGSTS B -> DEPBAR -> LDS B
v8: LDG A -> STS A -> LDGSTS B -> BAR -> LDS A -> 完整 sort -> DEPBAR -> LDS B
v9: CALL(issue B) -> LDG A -> 完整 sort -> CALL(wait B) -> LDS B
```

这三组对照同时回答“有没有重叠”和“扩大重叠是否值得”两个不同问题。

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

如果面试官问“为什么不用 inline asm memory clobber 就够了”，回答是：
compiler barrier 能限制编译器对某些内存操作的重排，但不一定建立目标数据
依赖，也不保证 ptxas 最终调度。v9 使用真实 noinline call boundary 是为了
做可证伪的调度实验，不是推荐在生产 Kernel 中普遍这样写。

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

本仓库当前 profile snapshot 中，几个最有解释力的对照是：

| 指标 | v0 brute-force | v7 optimized | 解释 |
|---|---:|---:|---|
| Kernel duration | 约 4.48 ms | 约 3.02 ms | profile 批次下的时间 |
| DRAM Throughput | 56.19% | 94.75% | 内存由未饱和变为接近上限 |
| Compute Throughput | 97.85% | 76.92% | 搜索执行压力显著下降 |
| No Eligible | 72.14% | 43.92% | 可发射性改善，但内存等待仍存在 |
| Eligible warps/scheduler | 1.43 | 2.48 | 调度器可选择工作增加 |
| Registers/thread | 30 | 27 | 没有用高寄存器代价换性能 |
| Achieved occupancy | 88.63% | 96.36% | 有足够 Warp 隐藏延迟 |

表中时间来自单次 NCU snapshot，简历 speedup 使用正常 benchmark 的五轮均值。
可以横向比较同一 profile 批次的指标，但不能把 snapshot 时间和其他批次的最快
时间拼起来计算加速比。

### 32. 你说的 934 GB/s 和 94.9% 是同一个指标吗？

不是。程序打印的约 934 GB/s 是按预先定义的 logical traffic 除以 Kernel
时间计算的“逻辑有效带宽”，适合版本间对比。NCU 的约 930.7 GB/s 和
94.75% 才是硬件计数器给出的实际 Memory/DRAM Throughput。

面试时应分别说明，不能把程序推算值冒充硬件实测值。

logical traffic 约为 2.819 GB，构成为 A 约 512 MiB、四份 B 约 2,048 MiB，
以及 count/checksum 两个结果数组约 128 MiB。`logical bytes / event time` 适合
同一数据契约下比较版本，但 cache hit、sector 重放和实际 DRAM bytes 只能由
硬件计数器回答。

### 33. 为什么 NCU 中的 Kernel 时间可能和 benchmark 不完全一致？

NCU 为采集计数器会 replay Kernel、控制时钟或引入 profiling 环境，时间不
一定等于正常运行。最终版本时间来自 CUDA Event、warmup 后重复执行的
benchmark；瓶颈归因来自 NCU。二者承担不同作用，不应混在同一组数字里。

每个可执行文件内部先 warmup 10 次，再用 CUDA Event 统计 50 次 launch；
`benchmark.sh` 默认再运行 5 轮并计算均值。`collect_profiles.sh` 使用 kernel
name 过滤，跳过前 10 次匹配 launch 后采集一个 timed launch。这样不会把
初始化或 warmup 混入目标 Kernel，但 profile replay 仍可能改变运行环境。

### 34. 1.38x 是怎样计算的？

以约 4.16 ms 的 Shared-broadcast brute-force baseline 和约 3.02 ms 的
优化版本为例：

```text
speedup = 4.16 / 3.02 ≈ 1.38x
time reduction = (4.16 - 3.02) / 4.16 ≈ 27.4%
```

“加速 1.38x”和“耗时降低约 27%”是同一结果的两种表达，不应说成性能
提升 38% 且耗时降低 38%。

仓库参考表里 shared baseline 是 `4.154 ms`，v7 是 `3.018 ms`：

```text
4.154 / 3.018 = 1.376 -> 1.38x
(4.154 - 3.018) / 4.154 = 27.35%
```

若改用 register baseline `4.208 ms`，则约为 1.39x。面试时必须先说清
baseline，不能为了得到更好的 speedup 临时切换分母。

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

还要检查三个容易被忽略的变量：编译目标统一为 `-O3 -arch=sm_89`；mainline
统一使用 `[group][B][element]`，不排除隐含 layout conversion；所有版本保持
相同 logical output。archive 中改变布局的实验只能作为设计探索，不能直接放入
mainline speedup 表。

### 39. 正确性是怎样验证的？

测试数据被确定性构造为每份 B 中 16 个元素存在于 A、16 个元素不存在。
Kernel 输出 survivor count 和 compact 后数据的 checksum，host 端检查所有
group 的 count 与 checksum，要求 `Wrong results: 0`。

checksum 不能替代生产环境的完整逐元素验证，但能防止编译器删除 compact
读写路径，并为 microbenchmark 提供低成本回归检查。

确定性输入中 A 是 32 个互异偶数的置换；每份 B 的偶数 lane 从 A 中选值，
奇数 lane 生成不在 A 中的值，因此期望正好 16 个 survivor。除了检查 count，
host 还按期望 survivor 求和并逐 group 比较 checksum。进一步工程化时应增加
随机输入、0%/100% duplicate、重复 key、边界整数以及逐元素 reference output。

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

duplicate ratio 还会改变 compact store、Shared Memory conflict 和后续消费的
工作量：0% 时 32 个元素全写，100% 时没有 compact payload，50% 只是中间点。
二分搜索本身轮数接近固定，但整个 Kernel 的瓶颈可能随 ratio 改变，所以不能
只凭 50% 结果宣称所有分布都获得同样 speedup。

### 42. 为什么 RTX 3090 上版本差距可能不同？

3090 与 4090 的显存带宽、缓存、SM 数量、时钟、调度器和 `cp.async` 实现
不同。一个在 4090 上已到 DRAM ceiling 的版本，在 3090 上可能更早受带宽
限制；也可能因为指令吞吐和 latency hiding 不同，使 ILP 或异步加载的收益
改变。因此不能直接移植绝对时间，应在目标 GPU 上重新编译、检查 SASS 和
采集 NCU。

迁移时应该按以下顺序重新判断：目标架构重新编译；确认 `int4`/`cp.async`
对应的 SASS；用相同数据契约跑 correctness；采集时间、DRAM、Compute、stall、
occupancy；最后再选择推荐版本。不能拿 4090 的 v7 排名直接当 3090 的结论。

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

## 硬件资源与底层追问

### 51. 你的 RTX 4090 有多少 Shared Memory？当前 Kernel 用了多少？

本机通过 `cudaGetDeviceProperties` 实测属性如下：

```text
GPU                         NVIDIA GeForce RTX 4090
Compute Capability          8.9
SM count                    128
Shared Memory / SM          102,400 bytes = 100 KiB
Default Shared / block      49,152 bytes  = 48 KiB
Opt-in Shared / block       101,376 bytes = 99 KiB
Registers / SM              65,536 x 32-bit
Max threads / SM            1,536 = 48 warps
Max blocks / SM             24
```

默认 512 threads/block，即 16 warps/block。v7 使用 17,152 bytes static shared，
v8 使用 19,200 bytes，都远低于 48 KiB per-block 默认上限。按 Shared Memory
计算 v7 每 SM 可以放 `floor(102400 / 17152) = 5` blocks，但线程上限只允许
`floor(1536 / 512) = 3` blocks，所以当前 occupancy 不是 Shared Memory 限制的。

### 52. Shared Memory 超了会发生什么？

需要区分三种“超”：

1. **未超过 per-block 硬上限，但每 block 用量变大。** Kernel 仍能 launch，
   但每 SM 可同时驻留的 block 数可能下降，occupancy 和 latency hiding 可能降低。
2. **超过默认 48 KiB，但没有正确使用 opt-in 配置。** 大容量 Shared Memory
   Kernel 不能按默认配置正常 launch。需要架构支持、使用适合的 dynamic shared
   配置，并通过 `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` opt in。
3. **超过该设备的 99 KiB per-block opt-in 上限。** launch 无法成立，会返回
   launch configuration/resource 类错误；这不是“自动溢出到 Global Memory”。

编译期可确定的静态 Shared Memory 如果明显超过可支持限制，可能由 `ptxas`
直接报错；运行期 dynamic shared 过大通常在 launch/error check 时暴露。因此
每次 launch 后都应检查 `cudaGetLastError`，同步后再检查异步执行错误。

### 53. Shared Memory 会像 CPU cache 一样自动换出吗？

不会。Shared Memory 是程序显式分配、显式寻址的 on-chip scratchpad。一个
block 的 Shared Memory 在该 block 整个驻留期间占用资源，不会因容量不足自动
spill 到 L2 或 DRAM。资源不足时是降低可驻留 block 数或 launch 失败，而不是
获得语义相同但更慢的透明后备存储。

### 54. 当前 occupancy 为什么接近 100%，怎样手算？

默认 block 有 512 threads，即 16 warps；4090 每 SM 最多 1,536 threads，即
48 warps。因此线程维度最多驻留 3 blocks，正好 48 warps，理论 occupancy 100%。

v7 的其他资源限制：

```text
Shared: floor(102400 / 17152) = 5 blocks
Regs:   floor(65536 / (27 * 512)) = 4 blocks（实际还受分配粒度影响）
Thread: floor(1536 / 512) = 3 blocks
```

最小值是 3 blocks，所以线程数是主要静态限制。NCU achieved occupancy 约
96.36%，低于理论值是运行时活跃情况和采样定义导致的正常差异。

### 55. 寄存器使用过多会怎样？和 Shared Memory 超限相同吗？

不完全相同。寄存器首先影响每 SM 能驻留多少 Warp/block；当编译器无法把 live
value 保存在分配到的寄存器中，还可能产生 local-memory spill。local memory
逻辑上是线程私有，但物理上位于 device memory/cache 层次，额外 `LDL/STL`
会增加延迟和流量。

当前 v7 是 27 registers/thread，NCU 没有显示 spill，资源计算也不是寄存器
受限。不能为了把 27 压到更低就盲目使用 `--maxrregcount`；若引入 spill，可能
用更高 occupancy 换来更差性能。正确方法是同时检查 registers、spill load/store、
occupancy 和最终时间。

### 56. Shared Memory bank conflict 到底怎样发生？broadcast 算冲突吗？

Ada 上可按 32 个 bank、连续 32-bit word 轮转映射来理解。一个 Warp 的同一条
Shared Memory 指令中，如果多个 lane 访问不同地址但落到同一 bank，请求通常
要拆成多个 wavefront；若多个 lane 读取完全相同的地址，则可以 broadcast，
不按普通多地址 bank conflict 处理。

所以 v0_shared 的“所有 lane 同轮读取同一个 A 地址”适合 broadcast；v6/v7
compact store 的 rank 各不相同，四个 subwarp 的动态地址可能落入相同 bank，
才出现 excessive wavefront。分析时必须区分 read broadcast 与 write conflict。

### 57. `__syncwarp` 和 `__syncthreads` 分别在这里保证什么？

`__syncwarp(mask)` 只同步 mask 中的 Warp lanes，并建立相应的 Warp 内存顺序；
compact 写入 Shared 后，读取同一 Warp 的 compact 结果前需要这个保证。

`__syncthreads()` 是整个 block 的 barrier。v5 以后各 Warp leader 先把输出写入
block Shared Memory，再由连续的 block threads 统一写 Global，因此跨 Warp
生产者和消费者之间必须 block 同步。v8 还用 block barrier 强制 A staging 的
调度边界，这也是它额外成本的来源。

场景题：若某些线程在 `__syncthreads()` 前提前 return，而同一 block 其他线程
到达 barrier，行为可能死锁或未定义。当前 kernel 的越界判断必须保证整个 block
对 barrier 路径一致；默认 group 数与 launch 也按完整 block 配置。

### 58. 为什么 Warp Shuffle 比 Shared Memory 适合保存 sorted A？

每个 lane 恰好持有一个 A，Bitonic compare-exchange 和二分 pivot 都是 Warp
内交换。Shuffle 可以直接在 lane register 间传值，不需要 STS/LDS、地址计算和
显式 Warp-level shared buffer。动态 source lane 的 `__shfl_sync` 又正好支持
每个 B 元素拥有不同的 `mid`。

代价是 Shuffle 也是指令，而且二分中的下一次 `mid` 依赖上一次比较，不能把
它当成零成本通信。Baseline 的问题正是 Shuffle/compare 轮数太多；优化的重点
是减少 stage，而不是简单地把所有 Shared Memory 都换成 Shuffle。

### 59. 为什么二分时四个本地 B 不能只做 5 次比较？

每个线程持有四个不同的 B 值，它们有四套独立的 `left/right/mid`。一条
`__shfl_sync` 指令每个 lane 可以选择一个 source lane，但同一 lane 的一次指令
只能返回一个 pivot 标量，不能同时为四个不同 `mid` 返回四个 pivot。因此四个
B 各自仍需要搜索轮次，源码中按 item 展开为四条搜索链。

可以尝试把四套搜索状态交错执行形成 ILP，但不会把信息论上的四次 lookup
合成一次。若想“一次取四个 pivot”，需要四条 Shuffle、向量化的多源 gather
能力，或把 A 放 Shared 后做四次 load，本质工作量仍在。

## 场景题与设计变体

### 60. 如果是 1A-2B，v6/v7 的 4x8 映射还能直接用吗？

逻辑结果可以做对，但直接沿用会有一半 lanes 没有有效 B，SIMT 和加载利用率
下降。更合理的候选方案有两种：

1. 一个 Warp 处理两个独立 group，每 16 lanes 负责一个 group；但 A 的排序
   需要完整 32-lane network，必须改成 16-lane 分区、每 lane 多持有 A，或先
   用完整 Warp 分阶段处理，设计复杂。
2. 一个 Warp 仍处理一个 group，让每份 B 用 16 lanes、每 lane `int2`，保持
   全 Warp 活跃；A 排序仍复用一次，但向量宽度和 compact 映射要重写。

选择不能只凭直觉，应为 N=2 单独实现 mapping，比较 A 排序摊销、lane 利用率、
load transaction、寄存器和时间。v7 是 N=4 的专用最优候选，不是通用答案。

### 61. 如果是 1A-1B、1A-8B，应该怎样映射？

1A-1B 时，最简单是一 Warp 每 lane 一个 B，加载天然连续，但 A sort 只摊销
一次，排序+二分未必胜过 brute force；也可让每 lane 处理多个 group 来增加
ILP，但会增加寄存器状态。

1A-8B 时，一个 Warp 可以分两批处理四份 B，排序后的 A 继续保存在寄存器，
排序成本摊销更充分；也可以一个 block 协作，但要注意 Shared Memory 与同步。
N 越大，sort-once 的算法优势越强，同时 B 流量更容易成为绝对瓶颈。

### 62. 如果列表长度小于 32 或不是 2 的幂怎么办？

小于 32 可以把无效 A lane padding 为哨兵值，并使用正确 active mask；Bitonic
network 可以按下一个 2 的幂执行。B 的 tail 也需要 mask，确保无效 lane 不参与
ballot、rank 和输出。

风险点是 `__shfl_sync` 的 mask：mask 中声明参与的 lane 必须实际执行对应
intrinsic，不能在部分 lane 随意 early return。哨兵还必须在 key 域外，或额外
携带 valid bit，避免与真实 `INT_MAX` 等边界值冲突。

### 63. 如果 A/B 长度大于 32，为什么不能简单循环？

可以循环，但原设计的核心性质会改变：A 不再能“一 lane 一元素全部驻留”，
排序网络需要多段寄存器或 Shared Memory，多 Warp 间还需要同步；二分 pivot
也不再能由单次 Warp Shuffle覆盖完整 A。

可选方案包括 block-level sort/search、A 分块并逐块 membership test、Shared
Memory hash table、或使用成熟的 block primitive。应根据长度、复用次数、key
类型和是否保持顺序重新选择算法，而不是把固定 32 Kernel 硬扩展。

### 64. 如果 B 指针没有 16-byte 对齐，你怎么处理？

在接口层声明并断言 alignment/stride contract；满足时走 `int4`/`cp.async 16B`
fast path，不满足时走标量或较窄向量 fallback。若只有首尾不对齐，可以用标量
处理 prefix/suffix，中间主体继续向量化。

性能测试必须把 fallback 纳入真实输入分布，不能在 benchmark 中依赖
`cudaMalloc` 对齐，集成后却接受任意 slice pointer。

### 65. 如果 B 中自己也有重复元素，当前算法会去掉吗？

当前 contract 只删除“存在于 A 的 B 元素”，不会对 B 内部再次 unique。
例如某个不在 A 的值在 B 中出现两次，两次都会被保留并按原顺序 compact。

如果需求改成 `B - A` 后还要 B 内去重，就需要额外定义保留第一个还是任意一个，
再增加 B 内 membership/排序或 hash 步骤。正确性 reference、checksum 和流量模型
都要更新，不能沿用当前性能数字。

### 66. 如果必须输出完整 compact 列表，3.02 ms 还能成立吗？

不能直接沿用。当前 Shared compact 被读取并归约成 checksum，只向 Global
写 count/checksum。完整输出会增加最多每 B 32 个 int 的 Global store，还需要
定义定长槽位还是变长全局数组。

定长槽位可以直接写 `[pair][32]`，简单但固定写入/预留大量空间；变长输出通常
需要两阶段 count + prefix sum + scatter，或 atomic 分配全局位置。无论哪种，
DRAM traffic、launch 数和瓶颈都会改变，必须建立新的 baseline 和 E2E 测量。

### 67. 如果 duplicate ratio 是 0%、50%、100%，你预期什么变化？

二分 membership 的轮数大体不变，但 compact 路径差别明显：

- 0%：32 个 survivor，Shared store/read 和 checksum 工作最多；
- 50%：当前测试，16 个 survivor；
- 100%：没有 payload store，count 为 0，compact 消费最少。

由于分支按固定每线程四个 item 展开，不同 valid pattern 还可能改变 predication、
bank conflict 和 Warp 执行效率。应测试 ratio 与 pattern 两个维度，例如相邻有效、
交错有效和随机有效，而不仅是平均比例。

### 68. 如果 A 已经有序，还需要 Bitonic Sort 吗？

不需要。若上游能保证每份 A 已排序且这个契约不会增加额外系统成本，可以直接
进入二分，省掉 15 个 Shuffle stage。此时应新增 sorted-A 版本，并把“上游排序
成本是否已经存在”纳入 E2E 评估。

不能在 microbenchmark 中免费预排序 A，再和需要处理无序 A 的 baseline 比，
否则改变了输入契约。只有真实调用方本来就产生有序 A 时，这才是公平优化。

### 69. 如果 key 从 32-bit int 改成 64-bit，会发生什么？

每份列表字节数翻倍，`int4` 映射要改成例如两个 64-bit key/16-byte load；
register、Shared Memory、比较和 Shuffle 宽度都可能增加。A/B traffic 翻倍后
更容易受 DRAM 限制，compact bank 映射也从每 key 一个 4-byte bank word变为
跨两个 bank 的访问。

算法仍可用，但资源占用、transaction 和 bank conflict 必须重新 profile，不能
按元素数不变就假设性能比例不变。

### 70. 如果要迁移到 H100 或更新架构，`cp.async` 还是最佳选择吗？

不一定。新架构可能提供更适合 tile 搬运的异步机制、不同 Shared Memory 容量、
Warp 调度和内存层次。迁移策略是先保留算法 contract 和 correctness，再用目标
架构推荐 primitive 重写搬运层，重新检查编译 SASS/机器指令与 NCU。

尤其不能因为 Ada 上 v7 推荐，就在所有架构上固定使用同一 inline PTX。inline
PTX 提高了控制力，也增加可移植和编译器兼容成本；生产代码应有架构 guard 和
fallback。

### 71. 如果换到 RTX 3090，为什么某个版本可能反而更快或更慢？

绝对性能由内存带宽、SM 数、时钟和缓存决定；版本相对排名还受 Shuffle/整数
吞吐、寄存器分配、occupancy、`cp.async` 调度和 ptxas 版本影响。3090 的同一
CUDA 源码可能生成不同 SASS，所以“4090 上扩大 overlap 无收益”不能机械外推。

面试中应提出可执行验证：用目标 `-arch` 重编译；锁定可比运行条件；运行多轮
benchmark；重新采 NCU；对关键 load/wait 做 SASS 定位；再报告目标卡结论。

## 工程与实验追问

### 72. 你为什么使用 CUDA Event，而不是 CPU `std::chrono`？

Kernel launch 对 CPU 是异步的，CPU wall clock 若没有正确同步，会主要测到
launch enqueue 时间；若每次都 `cudaDeviceSynchronize`，又会混入同步开销。
CUDA Event 记录在 GPU stream 时间线上，适合测一段 GPU 工作的 elapsed time。

事件前仍要 warmup，事件后要同步 stop event，并检查 launch/runtime error。
若要分析 CPU launch、同步和多 stream 的完整时间线，应使用 Nsight Systems
或 E2E wall time，而不是只看 Event。

### 73. Nsight Systems、Nsight Compute 和 SASS 各自回答什么？

- Nsight Systems：CPU-GPU 时间线、launch、memcpy、同步、stream 并发，回答
  “系统时间花在哪里”。
- Nsight Compute：单 Kernel 的硬件 counter、roofline、memory/workload、stall、
  occupancy，回答“这个 Kernel 为什么慢”。
- SASS：最终机器指令、load/store 宽度、barrier、wait 和调度顺序，回答“编译器
  实际让 GPU 执行了什么”。

三个工具层次不同。仅用 NCU 不能证明 CPU-GPU 同步减少；仅用 Systems 不能
证明 bank conflict；仅看 SASS 也不能证明某条指令变化最终提高了吞吐。

### 74. 你如何证明优化不是测量噪声？

对 1.38x 这种大差异，五轮均值和一致 correctness 通常足以建立主结论；对
v6/v7 的 0.05% 或 v7/v8 的 0.13%，必须更谨慎：交错版本运行、报告均值/标准差
或置信区间、监控温度/时钟/功耗、增加轮次，并复核 NCU/SASS 是否有符合假设
的变化。

若差异小于 run-to-run 波动，应表述为“性能持平”而不是宣布新版本更快。当前
`cp.async` 的价值更多是可复现的数据路径和调度研究，核心 1.38x 仍来自算法重构。

### 75. 为什么仓库 README 的稳定时间和 profiles 中某次 result 不一样？

README 表是某次受控条件下五轮 benchmark 的均值；profiles 中每个目录保存的
是后来某次 NCU 收集前生成的单次 result 和 profile snapshot。GPU boost、温度、
后台负载、驱动和采集日期都可能让绝对时间变化。

正确做法不是删掉不同数字，而是给它们标注用途：benchmark 均值用于 speedup，
同批 profile 用于瓶颈对照，SASS 用于指令证据。若正式对外发布新结果，应一次性
重新采集所有版本、记录环境，并同步更新 README，避免跨批次选择数字。

### 76. 公开仓库能证明简历里的哪些话，不能证明哪些话？

可以直接证明：固定 32/1:4 contract、brute-force baseline、Bitonic + Binary
Search、Warp Shuffle、Shared compact、4x8 subwarp、`int4`、`cp.async`、各版本
正确性与时间、NCU 指标、SASS 调度实验，以及约 4.16 -> 3.02 ms 的独立 Kernel
结论。

不能由仓库单独证明：内部调用链怎样接入、减少了多少 CPU-GPU 同步、真实数据
集上的完整流程 E2E、团队分工和线上约束。这些问题必须基于自己实际参与的内部
代码、实验日志和职责回答。最危险的说法是拿 microbenchmark 的 1.38x 去替代
完整系统 speedup，或暗示公开仓库复现了未公开的上层实现。

### 77. 面试官追问上层 E2E 加速，你应该怎样拆解？

先给公式而不是只报结果：

```text
T_total = T_CPU_prepare + T_H2D/D2H + T_sync + T_filter
        + T_other_kernels + T_framework_overhead
```

然后说明 Kernel 1.38x 只直接降低 `T_filter`；设备端过滤还可能减少 transfer、
sync 和中间 host work，因此 E2E 收益不一定等于 Kernel speedup。要用 timeline
展示优化前后的同步点和 memcpy，用相同数据集、参数、正确性和构建配置重复测试。

如果 E2E 收益大于 Kernel 1.38x，合理解释通常是同时消除了同步/传输或减少了
后续工作量；如果小于 1.38x，则符合 Amdahl's law，因为其他阶段没有加速。
所有具体倍数都应能指出数据集、规模、统计方式和 baseline commit。

### 78. 如果让你现在继续优化，你会怎样安排优先级？

1. 先补 workload matrix：duplicate ratio、N=1/2/4/8、不同 block size，确认
   推荐版本的适用域。
2. 针对 NCU 已定位的 compact store，尝试 bank-aware swizzle或两阶段 subwarp
   store；每个方案先看是否减少 excessive wavefront，再看时间。
3. 测完整输出 contract，判断真实场景是否因 Global store 改变瓶颈。
4. 到上层调用方做 Kernel fusion/设备端消费，优先减少全局字节、launch 和同步。
5. 只有在目标架构和 workload 改变后，再投入复杂 cp.async pipeline。

优先级依据是潜在收益上限：已经 95% DRAM 时，微调现有搬运很难得到大收益；
减少必须搬运的字节和边界开销更可能带来可见 E2E 改善。

### 79. 你怎么发现 CPU-GPU 迭代同步是上层瓶颈的？

这类结论不能由单 Kernel 的 NCU 得出。应先用系统时间线把一轮迭代拆成 CPU
准备、memcpy、Kernel、host synchronization 和下一轮依赖，观察 GPU 是否在
两轮之间存在空洞，以及 CPU 是否等待 device result 后才能继续发起工作。再用
host 侧分段计时和调用栈确认是哪一个结果触发同步。

可信证据应包括：优化前后相同输入下的时间线；同步/API 调用次数；H2D/D2H
字节；GPU active time 与 idle gap；完整流程正确性。面试中若没有带内部报告，
可以讲方法和自己实际观察到的模式，但不要伪造精确的 timeline 数字。

### 80. On-Device Candidate Filtering 的核心设计是什么？

核心不是简单把一段 CPU 代码翻译成 CUDA，而是改变中间结果的所有权：候选在
GPU 上生成后，直接在设备端完成 membership、去重/过滤和 compact，让后续 GPU
阶段消费，只把最终必要结果或控制信息交回 host。这样可能同时减少 D2H/H2D、
host filtering、同步点和下一轮 GPU 空闲。

实现上要处理四类问题：设备端 buffer 生命周期和容量；变长 compact 的位置
分配；跨 stream/Kernel 的依赖；与原 CPU reference 的逐元素正确性。这个公开
仓库只抽取了其中固定 32 的过滤 microkernel，不包含完整 buffer orchestration，
因此回答时要把 Kernel 技术与系统集成职责分开。

### 81. 为什么简历中的 E2E 是 1.76x-2.26x，可能大于 Kernel 的 1.38x？

因为两组数字优化的范围不同。Kernel 的 1.38x 只比较同一过滤 contract 的设备
执行；设备端集成还可能消除 CPU-GPU round trip、减少同步等待，并让后续阶段少
处理无效候选。所以 E2E 可能获得叠加收益，大于单 Kernel speedup并不矛盾。

不同数据集得到 1.76x 到 2.26x，通常应从候选规模、过滤率、迭代次数、原同步
占比和其他阶段占比解释。回答时至少准备一张按阶段拆分的 before/after 表，并
说明数据集、规模、参数、GPU、重复次数和 correctness。若拿不出这些条件，倍数
只是孤立数字，经不起追问。

## 算法正确性深挖

### 82. 你怎样证明 Bitonic Sort 后每个 lane 持有正确的有序元素？

32 元素 Bitonic network 是固定 compare-exchange 网络。外层 `k` 表示当前要
形成的 bitonic sequence 长度，内层 `j` 表示配对距离；每个 lane 通过
`lane ^ j` 找 partner，并根据 `(lane & k)` 决定升序或降序方向。每个 stage
所有 lane 同步执行一次 `shfl_xor` 和 min/max 选择。

正确性不能只靠“这是经典算法”。测试应覆盖随机排列、已升序、已降序、重复值、
全相等、`INT_MIN/INT_MAX`，并将 Warp 输出逐元素与 host `std::sort` 比较。
当前 benchmark 的 A 是互异偶数置换，足以验证主路径，但不是完整排序单测。

### 83. A 中存在重复值会影响二分和过滤正确性吗？

membership 语义只关心“是否至少存在一次”，所以 A 中重复值不会破坏二分搜索；
排序后相等值连续，命中任意一个即可返回 duplicated。它会减少 A 的有效集合大小，
但不会要求输出 A 的 unique 结果。

需要补测的是二分边界：目标小于最小值、大于最大值、等于首尾、位于重复区间。
若以后需求变成统计 multiplicity 或删除 B 中与 A 对应次数的元素，当前 bool
membership 就不够，需要 lower/upper bound 或计数结构。

### 84. 通用二分为什么使用 `[left, right)`，循环不变量是什么？

半开区间便于表达：候选位置始终位于 `[left, right)`；初始是 `[0, 32)`；若
`pivot < target`，令 `left = mid + 1`，否则收缩 `right = mid`，同时可以在
`pivot == target` 时记录命中。每轮区间严格缩小，最终 `left == right`。

面试时应能手推目标小于最小值和大于最大值两个边界，避免出现 `right=31` 与
半开区间混用、死循环或漏查 index 31。固定展开 6 轮是实现选择，不等于所有
32 元素二分理论上必须比较 6 次。

### 85. 二分搜索中的 Warp divergence 严重吗？

不同 B 值会产生不同的 `left/right/mid`，但代码主要用 predication/条件更新，
所有 lane 仍执行相同数量的展开轮次和 Shuffle。动态 `mid` 不代表动态 source
必须相同，Shuffle 允许每个 lane 选择不同 source lane。

如果使用“命中后立即 break”，不同 lane 的退出轮次可能引入 divergence，并且
Warp intrinsic 的参与 mask 更难保证正确。固定轮次虽然可能多做少量工作，但
执行结构稳定，适合这个固定 32 的 throughput Kernel。

### 86. Compact 为什么能够保持 B 的原始顺序？

每个元素的 logical position 是 `sub_lane * 4 + item`，与 B 的原始 index 一致。
对某个有效元素，rank 等于 `valid_mask` 中所有更小 logical position 的有效 bit
数量。因此若 `i < j` 且二者都有效，必有 `rank(i) < rank(j)`，写入 compact 后
顺序保持不变。

这也是 ballot/popcount 相比 unordered atomic append 的重要优势。若业务不要求
stable compact，可以考虑其他写法，但必须重新评估是否真的减少指令或冲突。

### 87. `valid_mask` 的构造为什么不会把四个 B 混在一起？

每个 8-lane subwarp 使用独立的 `subwarp_mask`，初始 4-bit local mask 左移
`sub_lane * 4`，再用宽度为 8 的 `__shfl_down_sync` 做 OR reduction。Shuffle
source 只在该 8-lane partition 内移动，因此最终每个 subwarp 得到自己 B 的
32-bit mask。

之后 `compact_base = (warp_id * 4 + b_id) * 33` 又给每份 B 分配独立 Shared
区域。面试时应同时说清“寄存器通信隔离”和“Shared 地址隔离”，只说一个不完整。

### 88. 为什么不用 CUB 的 BlockScan/WarpScan 做 compact？

CUB 是可靠的通用 primitive，生产代码应优先评估；但这里每份 B 固定 32 bit，
且每线程恰好四个连续元素，一个 32-bit mask 加 `popc` 就能同时得到总数和 rank，
状态更小，也自然保持顺序。引入通用 scan 可能增加临时存储、同步和模板生成代码。

正确的工程态度不是“手写一定更快”，而是用 CUB 版本作为可维护 baseline，比较
编译资源、SASS 和时间。若长度变长或每线程 items 数变化，CUB 的通用性可能更值。

### 89. 为什么 checksum 可以防止编译器把 compact 路径优化掉？

compact 结果随后从 Shared Memory 读取、归约并写到 Global output，host 又会
读取和校验 output，因此这条数据流具有可观察副作用。编译器不能在保持程序语义
的情况下删除整个 compact store/load 和 checksum。

但 checksum 不是无碰撞的正确性证明：不同错误排列或数值可能产生相同总和。
更强测试应输出完整列表，或组合 count、sum、xor/hash，并用小规模逐元素 reference
进行回归。性能 benchmark 使用 checksum 是成本与可观察性的折中。

### 90. 如果元素值溢出 `int`，checksum 会不会错误？

会有风险。当前确定性测试值较小，32-bit sum 不溢出；若 key 可覆盖完整 int
范围，32 个值相加可能超出 signed int。C++ signed overflow 的语义也不能随意
依赖。工程实现应使用 `int64_t/uint64_t` checksum，或明确采用无符号模加，并让
host reference 使用完全相同语义。

这不会改变 membership 本身，但会影响验证可靠性和额外寄存器/Shuffle 宽度，
修改后需要重新测资源和时间。

## CUDA 执行模型追问

### 91. 一个 block 为什么设为 512 threads，而不是 128 或 256？

512 threads 等于 16 warps，能让每 block 汇聚 64 个 B 结果后连续写回；在本机
4090 上每 SM 最多驻留 3 个这样的 block，理论上正好 48 warps，达到线程维度
100% occupancy。它在当前实测中表现稳定。

但 512 不是理论唯一最优。较小 block 可能减少 barrier 等待尾部、改善 block
调度或资源粒度；也会缩小 block-coalesced store 批次。应编译 128/256/512 的
参数化版本，比较 occupancy、waves、barrier stall 和时间，而不是只看 occupancy。

### 92. `__launch_bounds__(BLOCK_SIZE)` 有什么作用？

它向编译器声明该 Kernel 的最大 threads/block，帮助 ptxas 在寄存器分配与
occupancy 之间做决策，并允许编译器检查/优化针对该上限的资源使用。这里只有
一个参数，没有显式指定 min blocks per SM。

它不保证运行时一定达到某个 occupancy，也不会自动让 512-thread launch 最快。
若实际 launch threads 超过声明上限会违反约束；修改 `BLOCK_SIZE` 时必须统一
宏、launch 配置、Shared 数组大小和 profiling。

### 93. `__restrict__` 在这些指针上为什么有用？有什么前提？

`__restrict__` 告诉编译器在该作用域中这些指针指向的对象不会通过其他受限指针
别名访问，从而允许更积极地保留 load、重排指令和消除冗余访问。input A/B 与
output count/checksum 在当前分配中确实是独立 buffer。

前提是调用方必须遵守 no-alias contract。若让 output 与 input 重叠，行为不再
满足编译器假设，可能产生难以定位的错误。`restrict` 是语义承诺，不是无条件的
性能装饰。

### 94. 为什么代码里需要 `size_t` 计算 group/global index？

默认 group 数超过四百万，B 元素总数约五亿；目前仍在 32-bit 正数范围内，但
字节偏移和未来规模扩展很容易越界。使用 `size_t` 让地址乘法按 64-bit 进行，
避免中间表达式先以 32-bit 溢出后再转换。

代价是 64-bit 地址算术可能增加指令，因此也应尽量把 block/warp 内的小索引保留
为 int，只在全局地址层使用 `size_t`。正确性优先于省一条地址指令。

### 95. Warp-synchronous 编程为什么仍要写正确 mask 和同步？

现代 CUDA 不应依赖“同一 Warp 永远隐式锁步”的旧假设。Independent Thread
Scheduling 下，跨 lane 数据交换要通过带 mask 的 `_sync` intrinsic；Shared
Memory 的生产者/消费者需要合适的 `__syncwarp(mask)` 保证执行和内存顺序。

mask 必须准确表示所有参与线程，而且被 mask 指定的线程都要执行相同 collective。
固定满 Warp 路径可用 `0xffffffff`，8-lane subwarp 则使用对应的 8-bit lane mask。

### 96. 四个 subwarp 使用不同 mask 时，会不会让一个 Warp 真正并行执行四条指令？

不会把一个物理 Warp 变成四个独立 Warp scheduler 实体。32 lanes 仍共同发射
Warp instruction；subwarp mask/width 只是限制数据交换范围。当前设计的优势是
一条 Warp 指令同时服务四份 B 的对应工作，而不是四个 subwarp 各自拥有独立
instruction stream。

如果四个 subwarp 走不同控制分支，仍可能产生 divergence 和串行执行。设计中
尽量让它们执行同构流程，只让数据和 mask 不同。

### 97. `volatile Shared Memory` 在 v8 中解决了什么，为什么不应滥用？

v8 用 volatile Shared round-trip 建立编译器不能轻易消除的可观察内存依赖，
配合 block barrier 强制目标 SASS 调度顺序。它是为了验证“完整 overlap 是否有
收益”的实验手段。

`volatile` 主要约束编译器对该访问的优化，不等同于跨线程同步或完整 memory
fence；正确性仍依赖 barrier。滥用会增加 STS/LDS、限制优化并放大 Shared
资源，所以 v8 证明调度后变慢是合理结果。

### 98. inline PTX 使用 `memory` clobber 能保证硬件执行顺序吗？

`asm volatile` 和 memory clobber 主要约束前端/编译器对相关内存操作的删除与
重排，但 ptxas 仍会在满足依赖和 PTX 语义的前提下进行机器级调度。它不能代替
真实的数据依赖、barrier 或 async wait。

因此项目用 SASS 验证最终顺序，并通过 v8/v9 构造更强边界。回答时应区分
source compiler barrier、PTX memory model 和 SASS scheduler 三个层次。

### 99. 为什么 `cp.async.wait_group 0` 后不一定需要整个 block 的 `__syncthreads()`？

每个线程发射 copy 到自己的 `s_b4[tx]`，随后同一线程读取该位置；wait 保证该
线程之前提交的 async group 完成，不需要为了跨线程可见性做 block-wide barrier。
v7 的 B staging 没有“线程 A copy、线程 B consume”的交叉所有权。

若改成协作 tile，某些线程搬运、其他线程读取，就需要额外同步来保证所有生产者
完成和 Shared 可见。不能把当前同线程 ownership 的结论推广到任意 cp.async。

### 100. Kernel 中两个 `__syncthreads()` 的代价如何判断？

barrier 指令本身有成本，更重要的是 block 内最快 Warp 必须等待最慢 Warp，可能
出现 barrier stall。v5-v7 的输出 barrier 用于跨 Warp 合并 Global store，是
语义必要的；v8 额外 barrier 是调度实验成本。

判断不能只数源码 barrier 个数，应看 NCU barrier stall、到达时间不均衡、每
block Warp 数和最终时间。若移除 barrier，必须先重新设计所有权，例如每 Warp
直接 coalesced store，而不是牺牲正确性换时间。

## 内存系统与吞吐追问

### 101. 你说 DRAM 接近 95%，A/B 数据会不会其实大量命中 L2？

默认输入总量很大：A 约 512 MiB、B 约 2 GiB，明显超过 RTX 4090 的 72 MiB L2，
并且每 group 基本流式读取一次，跨 group 没有刻意复用，因此不能整体驻留 L2。
这与较高 DRAM Throughput 相符。

但不能只凭工作集大小断言所有 load miss。应看 L1/L2 hit rate、DRAM bytes 和
sector 指标。A 在同一 Warp 内的复用发生在寄存器排序结果上，而不是依赖跨 Warp
L2 reuse。

### 102. RTX 4090 理论显存带宽怎样估算？为什么和 NCU 的 peak 不一定相同？

按 384-bit 总线和约 21 Gbps GDDR6X 数据率，可估算理论带宽约：

```text
21 Gbit/s * 384 / 8 ~= 1008 GB/s
```

这是规格级理论值。实际 boost/memory clock、控制器效率、ECC/协议开销、读写
混合和 NCU 的 peak sustained 定义会让可持续上限不同。约 930 GB/s 对 1008
GB/s 是约 92%，而 NCU 报 94.75% 使用的是工具定义的 sustained peak 分母，
所以两个百分比不能混算。

### 103. Coalesced access 的判断单位是什么？是不是相邻线程相邻地址就结束了？

基本判断从同一 Warp memory instruction 的所有 lane 地址出发，硬件将地址请求
合并为若干 sector/cache-line transaction。相邻线程相邻地址通常有利，但还要看
访问宽度、对齐、活跃 mask、地址是否跨边界以及是否发生 replay。

v6 每 lane 16 bytes，整个 Warp 连续覆盖 512 bytes，是规则高利用率模式；它不
意味着一个 transaction。应以 requested bytes、sectors、transactions 和实际
throughput 验证合并质量。

### 104. 128-byte B 为什么还强调 16-byte 对齐，而不是只要 128-byte 对齐？

每线程执行的是 16-byte `int4`/`cp.async`，其最低契约是每个线程起始地址满足
16-byte 对齐。因为 B base 是 128-byte 对齐且 lane offset 为 `16 * sub_lane`，
自然满足更小的 16-byte 对齐。

128-byte base 对齐有利于 B 不跨越某些缓存边界，但是否必须取决于 allocator 和
layout；源码类型转换首先必须满足 C++/指令的 16-byte alignment contract。

### 105. v5 的 coalesced store 为什么提升很小？

输出只有每个 A-B pair 的 count 和 checksum，各 4 bytes，总流量约 128 MiB，
远小于 A+B 输入约 2.5 GiB。原来 store 虽稀疏，但不是总流量主体；合并后还新增
Shared 暂存和一次 block barrier。

所以 v5 的价值是改善输出 transaction 形态，整体时间只小幅变化符合 Amdahl's
law。若改成完整 compact Global output，store 占比上升，coalescing 的价值可能
明显不同。

### 106. 为什么减少指令后 DRAM 流量没变，时间却能下降到带宽上限？

Baseline 虽然读取相同数量的 Global A/B，但执行单元无法足够快地消费和发起
后续内存请求，DRAM 只有约 56% 利用率。减少搜索依赖后，Warp 更快推进，更多
memory request 能持续进入内存系统，DRAM 并行度和吞吐提升，总时间下降。

到达约 95% 后，即使再减少几条整数指令，必须搬运的字节仍需要约同样时间，
因此 v1 之后收益收敛。这就是“同样 bytes、不同 request generation rate”。

### 107. 这个 Kernel 的 arithmetic intensity 应该怎么算？

严格 roofline 通常使用 operation count / actual bytes。这里主要是整数比较、
Shuffle、地址和 compact，并非典型 FLOP Kernel，所以更适合报告 comparisons 或
warp stages per byte，同时用 NCU 的 Compute/Memory SOL 判断瓶颈。

Baseline 与优化版 Global logical bytes近似相同，但比较/stage 大幅减少，所以
operation intensity 下降。不要为了套 roofline 把所有指令粗暴当 FLOP；应明确
自定义 operation 的定义，或使用 instruction roofline/Speed of Light 分析。

### 108. 如果把 A 缓存在 Shared Memory 供多个 Warp 复用，会更快吗？

当前 contract 是每个 group 有自己的 A，一个 Warp 处理一个 group，不同 Warp
没有共享同一 A。把 A 放 block Shared 只会增加 STS/LDS 和容量，不能产生跨 Warp
复用；v0_shared 也说明 Shared broadcast 没有解决算法轮数。

如果业务 contract 改成多个 group 真正共享同一 A，block-level cache 才可能
有价值。这时需重新安排一个 block 内 group、同步一次加载，并比较 Shared
占用与减少的 Global load。

## Profiling 陷阱与实验答辩

### 109. NCU 的 Speed of Light 指标能直接相加吗？

不能。Compute Throughput 和 Memory/DRAM Throughput 是各自相对峰值的利用率，
不是互斥时间占比；例如一个 Kernel 可以同时有较高 compute 和 memory 指标。
它们不能相加成 100%，也不能仅凭谁大几个百分点就结束归因。

需要结合 workload analysis、scheduler、stall、instruction mix、memory table 和
版本对照。项目的瓶颈迁移可信，是因为多项指标与算法 stage 变化方向一致。

### 110. 面试官说“94.9% 是工具估算，不代表真带宽”，你怎么回答？

同意不能只靠一个百分比。我的证据包括 NCU 实际 Memory Throughput 约
930.7 GB/s、DRAM SOL 94.75%、大工作集流式访问、优化后版本时间收敛，以及
继续增加 ILP/完整 overlap 没有收益。logical effective BW 约 934 GB/s也与硬件
计数器数量级一致，但我明确不把它当同一个指标。

如果要进一步增强结论，我会导出 DRAM read/write bytes、memory clock、sector
计数，做独立 copy/stream bandwidth calibration，并在锁频条件下重复采集。

### 111. NCU replay 会不会改变 Kernel 行为？

NCU 为收集不能同时获得的 counter，可能多次 replay Kernel，并影响 cache state、
时钟和时间。因此 NCU duration 不作为最终 benchmark 数字，profiling input/output
必须可 replay，且不能依赖每次 launch 不可逆地改变状态。

项目 Kernel 对固定输入产生独立输出，适合 replay。最终速度使用无 NCU 的 CUDA
Event 多轮测试，NCU 用于同批版本的归因，这是两套职责。

### 112. 为什么只 profile 一次 launch，不 profile 50 次平均？

完整 metric set 对每个 launch 可能需要多 pass，profile 50 次会极慢并生成大量
重复数据。脚本先跳过 10 次 warmup，再采一个稳定状态的目标 launch，用于分析
指令和硬件 counter。

单 launch profile 不能承担统计结论，所以正常 benchmark 另做 50 次内部平均和
5 轮外层均值。若怀疑 profile 波动，应采多个独立 report，而不是在一个 report
里盲目增加 launch count。

### 113. 你怎样从 SASS 判断 `int4` 或 `cp.async` 真正生成了宽搬运？

直接 load 版本要定位对应 Global load 的操作数宽度和目标寄存器组；async 版本
定位 `LDGSTS`，并结合源码关联、地址步长和 16-byte PTX operand 判断每 lane
搬运宽度。还要检查编译器是否拆成多条窄指令、是否出现额外 STS/LDS。

单看 opcode 名不够，应把该指令前后的地址计算、dependency barrier 和 consumer
一起看，再用 NCU requested/actual sectors确认数据路径。

### 114. 为什么一个 NCU estimated speedup 不能直接当优化收益？

NCU suggestion 通常针对某一局部瓶颈，假设其他条件不变，并不包含修复该问题所
需的新指令、资源、同步或新的瓶颈。例如消除 Shared bank conflict 可能需要
swizzle、scan 或分阶段 store，额外成本可能抵消理论收益。

所以建议用于生成 hypothesis：实现独立版本，先确认目标 metric 改善，再以正常
benchmark 判断总时间。v8/v9 正是“局部 overlap 变好但整体更慢”的实例。

### 115. 如何确认没有测到初始化 Kernel 或其他 CUDA 操作？

CUDA Event 只包围目标 Kernel 的重复 launch；输入初始化在计时区间外完成。NCU
脚本使用精确 kernel-name filter，并通过 launch skip 跳过 warmup。结果文件还会
打印 variant label、problem size、时间和 correctness，防止跑错可执行文件。

更严格时可用 Nsight Systems 检查时间线，或用 NVTX range 标记目标阶段。不要
只看到一个 3 ms 数字就默认它来自正确 Kernel。

### 116. 版本很多，如何避免 benchmark 脚本把旧二进制当成新源码？

构建脚本应让每个 target 对应明确源码依赖，并在 benchmark 前执行增量或干净
构建；输出中打印版本标签。对发布数据，还应记录 git commit、`nvcc --version`、
driver、GPU、编译命令和二进制 hash。

本仓库 v7-v9 通过宏 include v6 主体，Makefile 必须把 `src/v6.cu` 作为依赖；
否则只改公共主体可能不会触发重编译。当前 Makefile 已显式设置这类依赖。

## 工程决策与行为面追问

### 117. 为什么保留 v0 到 v9，而不是只提交最快的 v7？

版本链展示每个 hypothesis 的因果证据：v1 证明算法重构是主要收益；v5/v6 验证
store 和 mapping；v7-v9 区分源码异步、真实 SASS overlap 和强制调度成本。
只有最快代码无法回答“为什么快”和“失败方案是否尝试过”。

工程上不一定把所有实验版本编进生产库，但研究仓库保留可复现版本和 profile，
有利于回归、跨 GPU 重新选择和面试答辩。

### 118. 如果团队要求删掉 v8/v9，你会怎样处理？

生产分支可以只保留推荐实现，降低维护面；但把 v8/v9 的结论、关键 diff、SASS
和 benchmark 留在实验文档或 archive。这样既满足产品代码简洁，也不会丢失“强制
overlap 为什么不划算”的知识。

决策依据应是维护成本、用户接口和回归风险，而不是对失败实验本身的否定。

### 119. 如果同事质疑 0.05% 的 `cp.async` 优化不值得合入，你怎么回答？

单从性能看，这个差异很可能接近噪声，不应把它作为主要成果。是否合入要比较
代码复杂度、架构兼容、维护成本和多轮统计。如果 inline PTX 增加明显维护风险，
v6 的直接 `int4` load 可能是更务实的生产选择。

项目保留 v7 是因为它是同 contract 下的推荐实验实现，并有完整 profiling；简历
的 1.38x 主要归功于算法重构。面对质疑应承认收益边界，而不是夸大千分级差异。

### 120. 你遇到的最大失败假设是什么，学到了什么？

失败假设是“只要把 `cp.async` 强制提到 A sort 前面，扩大 overlap 就会更快”。
v7 的 SASS 先证明源码顺序没有实现完整 overlap；v8/v9 又成功强制顺序，却分别
因 Shared round-trip/barrier 和 CALL/RET 变慢。

结论是性能优化必须分三步验证：想法在算法上是否合理；编译器是否生成目标机器
行为；机器行为是否改善最终瓶颈。前两步成立仍不保证第三步成立。

### 121. 如果面试官让你现场设计一个新优化实验，你怎么回答？

我会选择 compact bank conflict，因为 NCU 已有明确证据。先提出两个互斥版本：
bank-aware address swizzle；每个 subwarp 分时写入。保持输入、搜索和输出不变，
检查 correctness、Shared excessive wavefront、指令数、barrier stall 和时间。

预期管理也要说清：当前已接近 DRAM 上限，即使 conflict 指标下降，时间可能不变。
实验的成功标准不只是“某指标变绿”，而是正常 benchmark 在统计意义上改善。

### 122. 如果新版本快 2%，但只在 100% duplicate 下有效，要不要采用？

先看生产 workload 分布。如果 100% duplicate 是常见且重要路径，可以做运行时
specialization 或基于已有元数据选择版本；若极少发生，为 2% 增加分支和维护成本
可能不值得。还要确认选择成本不会吃掉收益。

报告时应写成“在 100% duplicate 场景快 2%”，不能概括成整体快 2%。性能结论
必须带适用域。

### 123. 如何向非 CUDA 面试官解释这个项目的核心价值？

可以把问题简化为：原方案对同一个 32 元素集合重复做全量扫描；我先付一次排序
成本，再让四组查询使用二分，把重复工作大幅减少。随后用硬件分析确认程序从
“算得不够快”变成“数据送得不够快”，最终接近机器带宽上限。

不要一开始堆叠 Shuffle、SASS、LDGSTS 等术语；先讲复用、算法复杂度、可测量
结果，再根据对方追问进入 CUDA 实现。

### 124. 如何回答“这个项目是不是为了写简历而做的 microbenchmark”？

应直接承认公开仓库是抽取后的独立 microbenchmark，它的价值是固定 contract、
隔离变量并公开复现 Kernel 机制；它不声称复现完整上层系统。简历中的系统集成
要由实际内部工作、timeline 和 E2E 数据回答。

抽取 microbenchmark 本身是合理性能工程方法，因为复杂系统中无法可靠归因每条
指令。但必须守住证据边界，不能让公开 benchmark 替代完整系统证据。

### 125. 如果面试官认为 1.38x 不够大，你怎么回答？

优化价值不能只看倍数，还要看热点占比、调用规模、正确性约束和接近硬件上限的
程度。这里在保持输入布局和输出语义下，将 Kernel 从执行受限推到约 95% DRAM
Throughput，后续局部优化收益收敛，说明 1.38x 有明确硬件依据。

同时不应防御性夸大：若完整系统中该 Kernel 占比低，Amdahl's law 会限制 E2E；
真正更大的系统收益来自减少同步和中间数据流。回答重点是“为什么这是可信且接近
当前约束上限的结果”。

### 126. 如果让你重新做一次，这个项目流程会怎样改进？

我会在最开始就建立测试矩阵和实验清单：N、duplicate ratio、block size、完整
输出/校验；统一锁频与环境记录；每个版本自动输出 JSON/CSV；CI 跑小规模
correctness，性能机定期跑 benchmark；profile report 与 commit 一一对应。

技术上仍先做算法重构，因为它贡献主要收益；但会更早加入 CUB/reference 版本、
完整随机正确性测试和 Nsight Systems 上层时间线，减少后期解释不同批次数据的
成本。

## 学习与模拟面试方案

### 第一阶段：能够白板讲清算法（1-2 天）

1. 手算 `4096` 次 baseline 比较和 `15 + 4 * 6 = 39` 个优化搜索 stage。
2. 在纸上画 32 lanes、4 个 8-lane subwarp、每 lane 四个元素的映射。
3. 用一个 8-bit 小例子手算 valid mask、`popc(lower_mask)` 和 stable rank。
4. 不看代码讲清 Bitonic 的 `5 * 6 / 2 = 15`，以及动态 source Shuffle 二分。

验收标准：能在 3 分钟内从输入 contract 讲到 compact 输出，并解释为什么不是
简单 `local_pos++`。

### 第二阶段：能够对着代码定位（2-3 天）

重点阅读：

- `src/v0.cu`、`src/v0_shared.cu`：两种 baseline；
- `src/v1.cu`：sort-once + binary search 的主收益；
- `src/v5.cu`：block-coalesced output；
- `src/v6.cu`：v6-v9 共用的 4x8、compact 和 async 主体；
- `src/v7.cu`、`src/v8.cu`、`src/v9.cu`：宏配置怎样生成三个实验。

练习：给面试官现场指出 `b_id/sub_lane`、`int4` 地址、valid mask、rank、barrier、
`cp.async commit/wait` 分别在哪，并手算 v7 的 17,152-byte Shared Memory。

### 第三阶段：能够用指标证明（2 天）

从 `profiles/v0` 和 `profiles/v7` 各挑一份 details，记住的不是所有指标，而是
证据链：`Duration -> Compute/DRAM -> Eligible/No Eligible -> occupancy/resource ->
shared conflict`。再从 `.sass` 中定位 `LDGSTS/DEPBAR/LDS`，比较 v7/v8/v9 顺序。

验收标准：面试官给出“DRAM 95%、No Eligible 44%”时，能说明为何不矛盾；
给出源码 `cp.async` 在前时，能立即回答必须检查 SASS。

### 第四阶段：准备场景题（1-2 天）

依次口述 1A-1B、1A-2B、1A-8B、长度非 32、64-bit key、未对齐输入、完整输出、
0%/100% duplicate 和迁移 3090/H100。每题固定按以下框架回答：

```text
先确认 contract -> 哪个原假设失效 -> 给出 2 个候选设计
-> 分析资源/流量/正确性 -> 设计 benchmark/NCU 验证 -> 不提前承诺收益
```

### 面试前必须能脱口而出的数字

| 数字 | 含义 |
|---|---|
| 32 | Warp lanes 和 A/B 固定长度 |
| 4 | 默认每份 A 对四份 B |
| 4,194,304 | groups |
| 16,777,216 | A-B pairs |
| 4,096 | baseline 元素比较/group |
| 15 | 32 元素 Bitonic stages |
| 39 | 约 `15 + 4 * 6` 个 Warp-level sort/search stages |
| 512 | threads/block，16 warps/block |
| 17,152 B | v7 static Shared Memory/block |
| 27 | v7 registers/thread |
| 4.16 -> 3.02 ms | shared baseline 到推荐版本 |
| 1.38x | `4.16 / 3.02` |
| 56% -> 94.9% | DRAM Throughput 瓶颈迁移 |
| 98% -> 77% | Compute Throughput 变化 |
| 0.05% | v6 到 v7 的微小收益，不能冒充主要加速来源 |

## 容易被追问的口径

- 不要说“v7 的 `cp.async` 完整隐藏了 B load”。SASS 证明其 overlap 很短。
- 不要把程序计算的 logical effective BW 当成 NCU 的真实 DRAM bandwidth。
- 不要说“达到 94.9% 后绝对无法再优化”，应限定 GPU、数据和输出语义。
- 不要说所有版本都改变了输入布局；当前 mainline 使用相同 B-major 布局。
- 不要把 Kernel microbenchmark 的 1.38x 直接说成完整应用的 E2E 1.38x。
- 不要只报最快时间，应说明 warmup、重复次数、正确性和 profiling 方法。
- 不要说“128-bit load 等于整个 Warp 只有一次 128-bit transaction”。
- 不要说 Shared Memory 超限后会自动 spill；它会限制驻留或导致 launch 失败。
- 不要把 v8/v9 变慢解释成 `cp.async` 无效；它们证明的是强制调度成本不划算。
- 不要把 4090 上的版本排名直接外推到 3090、H100 或不同 CUDA 工具链。
