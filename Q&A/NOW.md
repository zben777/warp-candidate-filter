# Warp Candidate Filter 面试问答

这份问答对应简历描述：

> 针对固定 32 元素候选列表，基于 Shared Memory、Warp Register/Shuffle 和 Binary Search 重构 Warp-level 1:N 去重内核；结合 Nsight Compute 和 SASS 定位 Warp 调度与访存瓶颈，通过 128-bit Vector Load、`cp.async` 持续优化数据路径。去重 Kernel 从 4.16 ms 降至 3.02 ms，加速约 1.38x；优化后 DRAM Throughput 约为峰值的 94.9%，主要瓶颈由 Warp/Search 执行迁移至 DRAM Bandwidth。

---

## 项目介绍

### 30 秒版本

> 我优化的是一个固定长度的 Warp-level 1:N 候选过滤 Kernel。场景是一份 reference list A[32] 需要被多份 candidate list B[32] 查询，判断 B 中哪些元素已经存在于 A。
>
> Baseline 对每个 B 元素都遍历整个 A，产生大量重复比较和 Warp-level Shuffle 依赖。针对固定 32 元素的特点，我首先利用 Warp Shuffle 在寄存器中完成 A 的 Bitonic Sort，将后续查询从线性扫描改成 Binary Search；随后进一步优化了 4×8 subwarp 映射以及 `int4` 128-bit Vector Load 和 `cp.async` 数据搬运。
>
> 最终在 RTX 4090 上 Kernel 时间从约 4.16 ms 降低到 3.02 ms，约 1.38×。通过 Nsight Compute 分析发现瓶颈由 Warp/Search 执行逐渐转移到 DRAM Bandwidth，优化版本 DRAM Throughput 达到约 94.9%。

### 一分钟版本

> 这个项目优化的是一个固定规模的 GPU 候选过滤 microkernel。每个 group 有一份 A[32] reference list，以及多份 B[32] candidate list。目标是在 GPU 上判断 B 中哪些元素已经存在于 A。
>
> 最初版本采用 Warp-level brute-force search，每个 B 元素都需要和 A 的 32 个元素比较。由于 A 会被多份 B 重复查询，存在大量重复比较和 Shuffle/compare 依赖。
>
> 针对固定 32 元素的特点，我设计了 sort-once + binary-search 的方案：首先利用 Warp Shuffle 在寄存器中完成 32 元素 Bitonic Sort，使 A 可以被多份 B 复用；随后通过 Shuffle 读取 pivot 进行二分搜索，将搜索阶段从 32 轮比较降低到约 5~6 轮。
>
> 在此基础上，我继续优化了 4×8 subwarp 数据映射以及 `int4` 128-bit Vector Load，并进一步探索了 `cp.async` 数据搬运。
>
> 最终 RTX 4090 上 Kernel 时间从约 4.16 ms 降低到约 3.02 ms，提升约 1.38×。通过 Nsight Compute 和 SASS 分析发现，优化前主要受 Warp/Search 执行限制，优化后 DRAM Throughput 提升到约 94.9%，瓶颈迁移到显存带宽。

---

## 核心问题

### 1. 你的这个项目是用来做什么的？

> 这个项目优化的是 GPU 上固定长度候选列表过滤（candidate filtering）问题。
>
> 输入是一份 reference list A[32] 和多份 candidate list B[32]。对于每份 B，需要判断其中哪些元素已经存在于 A。
>
> 由于单次列表长度只有 32 个元素，比较适合利用 CUDA Warp 进行专用化优化；但在大规模调用场景下，这类小操作会被重复执行很多次，因此目标是提高整体 GPU 吞吐。
>
> 因此我将这个操作抽象成独立 CUDA microbenchmark，在固定输入规模、数据布局和 launch 配置下，对不同 Warp-level 实现进行优化和分析。

**可能追问：为什么值得单独优化这么小的列表？**

> 虽然单次只有 32 个元素，但这类操作通常位于更大规模搜索或者候选生成流程中，调用次数非常高，因此整体吞吐会受到影响。
>
> 同时固定 32 元素天然对应一个 CUDA Warp，可以将一个列表映射到一个 Warp，避免通用数据结构的额外开销，因此适合设计专用 CUDA microkernel。

---

### 2. Baseline 是怎么做的，主要瓶颈在哪里？

> Baseline 采用 Warp-level brute-force search。一个 Warp 负责处理一个 group，其中每个 lane 持有一个 B 元素，需要判断该元素是否存在于 A[32]。
>
> 对于每个 B 元素，Kernel 会遍历 A 中的 32 个元素，通过 Warp Shuffle 或 Shared Memory broadcast 将 A 的元素依次提供给整个 Warp，然后进行比较。
>
> 默认一个 group 有四份 B，因此一次处理需要执行：
>
> `4 × 32 × 32 = 4096`
>
> 次元素级比较。
>
> 这种方法的问题不是简单的访存不合并，而是大量重复搜索导致的 Warp-level Shuffle、整数比较以及串行依赖。
>
> 从 NCU 分析来看，register/shuffle baseline 中 Compute Throughput 约 97.9%，DRAM Throughput 约 56.2%，说明 Kernel 主要受 Warp/Search 执行和指令依赖限制，而不是显存带宽限制。

**可能追问：为什么 No Eligible 高却不是内存瓶颈？**

> `No Eligible Warps` 表示某些周期 scheduler 没有可发射 Warp，但它本身不能直接说明原因。
>
> 在 baseline 中，Compute Throughput 接近 98%，DRAM Throughput 只有约 56%，同时 SASS 中存在大量 Shuffle 和 compare 的串行依赖，因此主要问题是执行 pipeline 中的依赖限制，而不是 DRAM 吞吐不足。

---

### 3. 你是怎么改进的？

> 我的优化主要分为两个阶段：首先通过算法重构降低搜索复杂度，然后进一步优化 Warp 数据映射和访存路径。
>
> 第一阶段是核心优化。由于 A[32] 会被多份 B 重复查询，我没有让每个 B 元素独立扫描 A，而是利用固定 32 元素的特点，让一个 Warp 中每个 lane 保存一个 A 元素，并通过 Warp Shuffle 完成 32 元素 Bitonic Sort。排序后的 A 可以被多份 B 复用。
>
> 在查询阶段，将原来的 brute-force search 改成 binary search。每个 B 元素通过动态 `shfl` 获取对应 pivot，在排序后的 A 上进行查找，将搜索阶段从 32 轮比较降低到约 5~6 轮。
>
> 第二阶段是在算法优化基础上的 CUDA kernel 调优。针对 1:N 的数据复用模式，我设计了 4×8 subwarp 映射，让一个 Warp 同时处理四份 B；同时通过 `int4` 进行 128-bit Vector Load，提高连续数据搬运效率，并进一步探索 `cp.async` 优化 Global Memory 到 Shared Memory 的数据路径。
>
> 这些优化分别降低了搜索阶段的执行压力和数据搬运开销。最终通过 Nsight Compute 和 SASS 分析验证，Kernel 瓶颈从 Warp/Search 执行逐渐迁移到 DRAM Bandwidth。

**可能追问：你如何确定每一步优化有效？**

> 每个优化思路都单独建立版本，通过相同输入、相同 launch 配置进行 benchmark。同时使用 Nsight Compute 分析性能指标变化，并通过 SASS 确认编译器是否生成了预期指令。

---

### 4. 你怎么判断性能已经接近瓶颈？

> 我主要从三个方面判断，而不是只看 Kernel runtime。
>
> 第一，看 Nsight Compute 的硬件指标。优化版本 DRAM Throughput 达到约 94.9%，说明 Kernel 已经接近当前 GPU 的显存带宽上限。
>
> 第二，看瓶颈是否发生迁移。Baseline 中 Compute Throughput 接近 98%，而 DRAM Throughput 只有约 56%，优化后 DRAM Throughput 提升到约 94.9%，说明搜索阶段优化释放了计算压力，瓶颈转移到了显存带宽。
>
> 第三，看后续优化的边际收益。Binary Search 后继续尝试 ILP、Vector Load、`cp.async` 等优化时，性能提升逐渐收敛。
>
> 因此在当前数据规模、输入布局和 RTX 4090 上，该 Kernel 已经从 execution bound 转变为 memory bandwidth bound。

**可能追问：95% 为什么不是 100%？**

> 理论峰值带宽通常是在特定访问模式下测得，真实 Kernel 还包含地址计算、指令执行、同步以及读写混合等额外开销，因此很难达到绝对 100%。
>
> 94.9% 说明当前数据路径已经接近硬件上限，继续优化已有的数据搬运收益有限。如果希望继续提升，需要减少 Global Memory traffic 或进行更高层的 Kernel 融合。

---

## 算法与映射

### 5. 这里的 Warp-level 1:N 是什么意思？

> 这里的 1:N 指的是 list 级别的复用关系。
>
> 其中 1 表示一份 reference list A[32]，N 表示多份 candidate list B[32]。一个 A 会被多份 B 查询，因此 A 的排序结果可以被重复利用。
>
> 在当前 benchmark 中，默认是 1 个 A 对应 4 份 B，也就是同一个 Warp 先处理一份 A，然后利用排序后的结果完成多份 B 的 membership search。
>
> 它不是指一个元素和 N 个元素比较，而是一份 reference list 对多个 candidate list 的复用关系。这也是为什么可以通过 sort-once + binary-search 降低整体搜索成本。

**可能追问：为什么不让每个 B 独立处理？**

> 如果每份 B 都独立搜索 A，那么每次都需要重复完成 A 的加载和搜索过程。
>
> 当前场景中 A 会被多份 B 查询，因此先对 A 做一次 Warp-level Sort，可以摊销排序成本，提高数据复用率。
>
> 这种设计利用了固定 32 元素的特点，让排序结果保存在 Warp register 中，通过 Shuffle 提供给后续查询。

---

### 6. 为什么固定为 32 个元素？

> 这里固定 32 个元素主要是为了匹配 CUDA Warp 的执行模型。
>
> 一个 Warp 正好包含 32 个线程，因此对于 A[32] 或 B[32] 这种固定长度列表，可以天然采用 one lane one element 的映射方式：
>
> lane0 处理第 0 个元素，lane1 处理第 1 个元素，直到 lane31。
>
> 这样每个线程只需要在寄存器中保存自己的数据，通过 Warp Shuffle 完成线程之间的数据交换，不需要额外的 Shared Memory 和 Block-level synchronization。
>
> 例如在 A 的 Bitonic Sort 中，每个 lane 保存一个元素，通过 `shfl_xor` 完成不同 stage 的 compare-exchange；在 Binary Search 中，也可以通过 Shuffle 广播 pivot 给 Warp 中的线程。
>
> 如果列表长度不是 32，比如 100 个元素，就需要多个 Warp 协同处理，会引入跨 Warp 通信、同步以及更复杂的数据布局问题。因此固定 32 元素可以充分利用 Warp-level primitive，设计更加专用化的 CUDA kernel。

**可能追问：为什么不用 Shared Memory 存储 A，而选择 Register + Shuffle？**

> 因为数据规模固定且只有 32 个元素，每个线程只需要保存一个元素，Register 可以直接提供低延迟访问。
>
> 如果使用 Shared Memory，需要额外的数据搬运和同步：
>
> `Register → Shared Memory → Register`
>
> 而 Warp Shuffle 可以直接完成：
>
> `Register → Register`
>
> 减少了共享内存访问和同步开销，更适合这种固定规模的小数据操作。

---

### 7. 一个 Warp 具体怎样处理一组数据？

> 一个 Warp 处理一组数据时，核心思想是利用 32 个线程和 32 个元素之间的一一对应关系，让数据尽量保存在寄存器中，并通过 Warp-level primitive 完成通信。
>
> 对于 reference list A[32]，一个 Warp 中每个 lane 保存一个元素：
>
> lane0 → A[0]  
> lane1 → A[1]  
> ...  
> lane31 → A[31]
>
> 每个线程将自己的 A 元素保存在 register 中，然后通过 Warp Shuffle 完成 32 元素 Bitonic Sort。排序完成后，整个 Warp 共享一个有序的 A。
>
> 对于 candidate list B[32]，由于一个 A 会被多份 B 查询，因此后续查询阶段可以利用排序后的 A 进行 Binary Search。
>
> 在进一步优化中，我将一个 Warp 划分为 4 个 8-thread subwarp：
>
> lane 0-7 → B0  
> lane 8-15 → B1  
> lane 16-23 → B2  
> lane 24-31 → B3
>
> 每个 subwarp 负责一份 B[32] 的查询任务，使一个 Warp 可以同时处理多个 candidate list，提高 A 的复用率。

**可能追问：为什么一个 Warp 不直接处理一份 B，而要拆成 4 个 subwarp？**

> 如果一个 Warp 只处理一份 B，那么 A 的排序成本只能服务这一份 B。
>
> 当前 1:4 场景中，把一个 Warp 划成 4 个 8-thread subwarp，可以让排序后的 A 同时服务四份 B，提高 A 排序结果的复用率，同时保持 Warp 内通信。

---

### 8. 为什么不让一个 Warp 只处理一份 B？

> 一个 Warp 只处理一份 B 是一种直接的映射方式，但在当前场景下不是最优。
>
> 因为 A[32] 已经完成排序，后续查询阶段主要是 Binary Search，单个 B 的计算量已经降低。如果一个 Warp 完全负责一份 B，会降低 A 排序结果的复用效率。
>
> 因此优化版本将一个 Warp 划分为多个 8-thread subwarp：
>
> lane 0-7 → B0  
> lane 8-15 → B1  
> lane 16-23 → B2  
> lane 24-31 → B3
>
> 这样同一个 Warp 可以同时处理多份 B，并共享已经排序好的 A。

**可能追问：为什么选择 8-thread subwarp？**

> 主要是根据当前 workload 选择。B 固定为 32 个元素，而查询阶段已经从线性扫描降低为 Binary Search，不再需要完整 Warp 的并行。
>
> 8-thread subwarp 可以在保持 Warp-level communication 的同时，让一个 Warp 同时处理 4 份 B，提高整体吞吐。

---

### 9. 32 元素 Bitonic Sort 为什么是 15 个 stage？

> 因为当前排序规模固定为 32 个元素，而 32 = 2^5。
>
> Bitonic Sort 的 stage 数量为：
>
> `log2(N) × (log2(N)+1) / 2`
>
> 对于 N=32：
>
> `5 × 6 / 2 = 15`
>
> 这 15 个 stage 包含多个 bitonic merge 阶段：
>
> `1 + 2 + 3 + 4 + 5 = 15`
>
> 在 CUDA 实现中，一个 Warp 的 32 个 lane 分别保存一个元素，每个 stage 通过 `__shfl_xor_sync()` 获取 partner lane 的数据，然后完成 compare-exchange。
>
> 由于规模固定，整个排序过程可以展开，不需要 Shared Memory 和 Block-level synchronization，非常适合 Warp-level optimization。

**可能追问：为什么不用 Shared Memory 做排序？**

> 因为排序规模固定为一个 Warp 的 32 个元素，每个 lane 本身可以保存一个元素。Warp Shuffle 可以直接完成 Register 到 Register 的交换，避免 Shared Memory 访问和同步开销。

---

### 10. 为什么排序加二分会比直接比较快？

> 核心原因是利用了 A 被多份 B 重复查询的特点，将一次性的排序成本摊销到多次查询中。
>
> Baseline 中，每个 B 元素都需要遍历整个 A[32]，一个 group 有 4 份 B 时，一次处理需要执行：
>
> `4 × 32 × 32 = 4096`
>
> 次元素级比较。
>
> 优化后，先利用 Warp Shuffle 对 A[32] 做一次 Bitonic Sort，使 A 变成有序序列。之后每个 B 元素在有序 A 上进行 Binary Search。
>
> 对于长度为 32 的数组，Binary Search 每次最多只需要：
>
> `log2(32) ≈ 5`
>
> 次比较。
>
> 因此 sort-once + binary-search 可以显著减少搜索阶段的执行压力，尤其适合一份 A 服务多份 B 的 1:N 场景。

**可能追问：排序本身不是有额外开销吗？为什么不会抵消收益？**

> 如果只查询一次 B，排序成本确实可能无法完全抵消 Binary Search 带来的收益。
>
> 我也测试过 1:1 的场景，即一份 A 只对应一份 B。在这种情况下，由于排序成本无法被多个查询摊销，sort-once + binary-search 和直接比较的性能差距并不明显，两者时间基本接近。
>
> 但是当前优化目标是 1:N 场景，一份 A 会被多份 B 重复查询。因此 A 的排序只需要执行一次，而查询阶段会执行多次，排序成本可以被充分摊销。
>
> 这个实验验证了该优化主要针对存在数据复用的 1:N candidate filtering 场景，而不是所有单次查询场景。

---

### 11. 为什么代码中的通用二分是 6 轮，而不是 5 轮？

> 从理论复杂度来看，32 个元素的 Binary Search 满足：
>
> `log2(32)=5`
>
> 也就是说，如果只考虑定位搜索区间，最多需要 5 次范围缩小。
>
> 但是代码中的实现目标是判断 candidate 是否存在，而不是只找到插入位置。因此在缩小搜索范围后，还需要一次最终的元素比较确认：
>
> `5 次范围缩小 + 1 次最终判断 = 6 轮`
>
> 同时 GPU kernel 中采用固定轮数的实现，而不是动态 while loop，这样可以保证 Warp 内不同线程执行相同次数的搜索步骤，减少控制流 divergence，也更方便编译器进行展开优化。

**可能追问：为什么不直接用 5 轮，然后最后统一检查？**

> 也可以这样设计，但是需要额外保存最终候选位置，并增加一次统一判断逻辑。
>
> 当前实现把搜索过程和存在性判断融合在固定 6 轮中，控制流更加简单，也更容易保持 Warp 内同步执行。

---

### 12. 二分搜索怎样读取其他 lane 持有的 A？

> 在优化版本中，A[32] 不存放在 Shared Memory，而是每个 lane 在 register 中保存一个元素：
>
> lane0 → A[0]  
> lane1 → A[1]  
> ...  
> lane31 → A[31]
>
> 因此 Binary Search 过程中，如果某一轮需要比较 A[mid]，当前线程并不能直接访问其他 lane 的 register。
>
> 这里利用 Warp Shuffle，通过 `__shfl_sync()` 直接读取指定 lane 的寄存器值。
>
> 例如当前需要比较 A[16]：
>
> ```cpp
> pivot = __shfl_sync(mask, value, 16);
> ```
>
> 表示从 lane16 的 register 中取出 A[16]。后续每一轮 Binary Search 都根据新的 mid，通过 Shuffle 获取对应 lane 保存的 A 元素。
>
> 这种方式实现了 `Register → Register` 的数据交换，不需要经过 Shared Memory。

**可能追问：为什么不用 Shared Memory 存 A？**

> 因为 A 只有 32 个元素，一个 Warp 中每个 lane 保存一个元素即可。
>
> Warp Shuffle 可以直接完成 Register 到 Register 的交换，避免额外的 Shared Memory load/store 和同步。

---

### 13. 为什么不使用 Hash Table 或 Bloom Filter？

> 这两种方法都可以解决 membership query 问题，但是当前场景具有固定规模和 Warp-level 执行特点，因此 Binary Search 更适合。
>
> Hash Table 更适合大规模动态集合查询，但是当前 A 固定只有 32 个元素。如果建立 Hash Table，需要额外维护 bucket、hash value 以及冲突处理逻辑，这些额外开销对于 32 个元素的小规模数据并不划算。
>
> 同时 GPU 上 Hash lookup 通常会产生更随机的访问模式，不容易利用当前 Warp Shuffle 的寄存器数据路径。
>
> 相比之下，当前方案先对 A 做一次 Warp-level Sort，然后 Binary Search。由于数据规模固定为 32，搜索次数固定为 5~6 轮，并且 pivot 可以通过 `__shfl_sync()` 在 register 之间交换，更符合 Warp-level execution。
>
> Bloom Filter 则存在 false positive，不能直接满足当前精确 membership query 的需求。
>
> 因此在固定 32 元素、只读、多次查询的 1:N 场景下，sort-once + binary-search 更合适。

**可能追问：如果 A 变成几千甚至几万个元素，还会选择 Binary Search 吗？**

> 不一定。当前选择 Binary Search 的原因是它匹配固定 32 元素和 Warp-level 专用化场景。数据规模变大以后，需要重新评估 Hash、分块搜索或者其他数据结构。

---

### 14. 为什么不把 B 也排序，然后做 merge？

> 如果只考虑一次 A 和一次 B 的比较，把两个列表都排序后进行 merge 确实是一种可行方案。
>
> 但是当前场景是 1:N 查询模式，一份 A 会被多份 B 重复查询。
>
> 如果对 B 排序，那么每份 B 都需要额外执行一次排序，这些排序成本无法复用。
>
> 而当前方案只需要对 A 做一次 Warp-level Sort，后续所有 B 都复用排序后的 A，通过 Binary Search 完成查询。
>
> 另外，Merge 更适合两个有序序列的一次性合并，而当前 workload 的核心是一个固定 reference list 对多个 candidate list 查询。因此 sort-once + binary-search 更符合当前复用模式。

**可能追问：如果场景变成 1:1，Merge 会不会更好？**

> 有可能。1:1 场景下排序成本无法通过多份 B 摊销，需要重新比较 brute force、sort+binary-search 和 sort+merge 的整体成本。

---

## Compact 与 Shared Memory

### 15. 去重后的 compact 是怎么做的？

> 去重后的 compact 主要是在 Warp/subwarp 内完成。
>
> membership check 之后，每个线程会为自己处理的 B 元素生成 valid bit。一个 8-lane subwarp 最终对应一份 B[32] 的 32 个 valid bit。
>
> 首先把这些 valid bit 合并成一个 32-bit `valid_mask`，`popc(valid_mask)` 可以得到 survivor 总数。
>
> 对于每个有效元素，再计算它之前有多少个 valid bit：
>
> `rank = popcount(valid bits before current element)`
>
> 这样每个有效元素都能得到唯一、并且保持原始 B 顺序的 compact 位置，最后写入 Shared Memory。
>
> 整个过程不需要 global atomic。

**可能追问：为什么不用 atomicAdd 给每个元素分配输出位置？**

> 因为固定 32 元素时，可以直接通过 bit mask + popcount 算出 rank。使用 atomic 会引入竞争和串行化，没有必要。

---

### 16. 为什么不能简单使用 `local_pos++`？

> 因为 `local_pos` 是线程私有状态，它只能知道当前线程自己已经处理了多少个有效元素，不知道前面其他线程找到了多少个 survivor。
>
> 多个线程如果都从 `local_pos=0` 开始，就可能计算出相同输出位置，造成覆盖。
>
> Compact 本质上需要的是跨线程 prefix rank，因此当前实现通过 valid mask + popcount 计算：
>
> `当前元素之前一共有多少个有效元素`
>
> 从而得到唯一 compact 位置。
>
> 固定 32 元素时，这种 Warp/subwarp 内的 mask + popcount 比 atomic 更适合，也天然保持 B 原顺序。

**可能追问：为什么不用 atomicAdd？**

> atomicAdd 可以实现，但会让多个线程竞争同一个 counter，引入不必要的 serialization。固定 32 元素时，mask + popcount 更轻量。

---

### 17. Shared Memory 在这个实现中负责什么？

> A 的排序和搜索主要在寄存器和 Warp Shuffle 中完成，Shared Memory 不是用来保存 sorted A 的。
>
> 在当前优化路径中，Shared Memory 主要承担三类工作：
>
> 1. 保存 compact 后的 B，供后续 Kernel 内消费；
> 2. 作为 `cp.async` 搬运 B 的落点；
> 3. 在 block-coalesced output store 中暂存 count/checksum，再由连续线程统一写回 Global Memory。
>
> 所以这个项目不是“把数据全部放进 Shared Memory”，核心搜索状态仍然在 Warp registers 中，而 Shared Memory 主要负责局部 staging 和跨线程的数据组织。

**可能追问：为什么 sorted A 不直接放 Shared Memory？**

> 因为 A 固定只有 32 个元素，一个 Warp 每 lane 保存一个元素即可。排序和二分 pivot 都是 Warp 内交换，用 Shuffle 可以直接完成 Register-to-Register 通信，没有必要增加 STS/LDS 和同步。

---

### 18. NCU 不是仍然报告了 Shared Memory bank conflict 吗？

> 是的，优化版本在 compact store 上仍然存在 Shared Memory bank conflict。
>
> 这个 conflict 主要来自四个 8-lane subwarp 同时对四份 B 做 rank-based compact store。虽然每份 compact list 做了 padding，但不同 subwarp 的动态 rank 仍然可能映射到相同 bank。
>
> 所以 NCU 会看到 excessive shared wavefront。
>
> 但是这里要区分“存在局部 bank conflict”和“它是不是当前主要瓶颈”。当前优化版本 DRAM Throughput 已经接近 95%，因此不能只看到 NCU 的 bank conflict warning 就直接认为它是最主要问题。
>
> 真正判断时需要把 Source/SASS 中对应的 `STS`、excessive wavefront、runtime 和整体 DRAM/Compute 指标结合起来看。

**可能追问：如果要继续优化这个 conflict，你会怎么做？**

> 可以尝试 bank-aware swizzle、进一步调整 stride 或分阶段写入，但这些方案本身也会增加地址计算或同步，所以最终还是要以实际 runtime 判断是否值得。

---

### 19. 为什么 Shared Memory stride 使用 33 而不是 32？

> 这里的 33 是为了让不同 compact list 的 Shared Memory 起始 bank 发生偏移，缓解规则性的 bank conflict。
>
> 对 32-bit 数据，可以近似按照：
>
> `bank = word_address % 32`
>
> 来理解。
>
> 如果 stride=32，那么相邻 list 的 base 相差正好 32 个 word：
>
> `32 % 32 = 0`
>
> 所以每份 list 的起始地址会落到相同 bank pattern。
>
> 改成 stride=33 后：
>
> `33 % 32 = 1`
>
> 相邻 list 的 base 会依次偏移一个 bank，从而打破完全相同的 bank 映射。
>
> 需要注意的是，33 只能缓解规则冲突，并不能保证完全 conflict-free，因为 compact rank 是运行时根据 valid mask 决定的，不同 subwarp 仍然可能动态落到同一 bank。

**可能追问：为什么不是 34？**

> 目标只是打破 stride=32 和 32 个 bank 的周期关系，33 是最小 padding，额外 Shared Memory 开销也最小。

---

### 20. v5 的 block-coalesced output store 做了什么？

> v5 优化的是 count/checksum 的 Global Memory 写回方式。
>
> 原来的路径是每个 subwarp 的 leader 直接写 global count/checksum，这样一个 Warp 里只有少数 lane 活跃，而且这些 lane 的写地址比较分散，store coalescing 不理想。
>
> v5 改成先让各个 leader 把结果写入 block Shared Memory，然后做一次 block 同步，再由连续的 block threads 把这些结果连续写回 Global Memory。
>
> 数据路径从：
>
> `少数 leader → sparse global store`
>
> 变成：
>
> `leader → Shared Memory staging → 连续 block threads → coalesced global store`
>
> 这样可以把稀疏的 global store 转换成更规则的连续写回，提高 store transaction 的有效利用率。
>
> 这一步属于 Binary Search 优化之后的数据路径微调，不是 1.38× 加速的主要来源。

**可能追问：为什么不能继续让 leader 直接写？**

> 逻辑上当然可以，v5 的目的只是减少少数 active lane 的分散 global store。是否值得取决于 staging 和同步成本，因此需要通过 benchmark 验证。

---

## 向量加载与数据布局

### 21. 128-bit Vector Load 是怎样实现的？

> 这里的 128-bit Vector Load 具体对应主线 v6 的 B 读取。输入仍是 `[group][B][element]`，一份 B 有 32 个 int，共 128 Bytes。一个 Warp 分成四个 8-lane subwarp，每个 lane 持有连续四个元素，用一个 `int4` 读取 16 Bytes，因此八个线程正好覆盖一份 B，整个 Warp 覆盖四份 B 的连续 512 Bytes。
>
> 源码在 `src/v6.cu` 中把 B 指针转换成 `const int4*`，再通过 `input_b4 + group_id * 32 + lane` 定位当前线程的数据。这既与 B-major 布局兼容，也让每个线程减少独立标量 load 和地址计算。包里的 v6 SASS 确实有 `LDG.E.128.CONSTANT`，说明这次编译生成了宽加载。
>
> 这里还要分清版本：v6 是 Global 到寄存器的直接 int4 load；v7 使用同一映射，但通过 16-byte cp.async 先把 B 搬到 Shared Memory，再用 `LDS.128` 读进寄存器。“每线程 128-bit”描述的是指令宽度，不代表整个 Warp 只有一次内存事务。

**可能追问：128-bit Vector Load 和 coalesced access 是同一件事吗？**

> 不是。向量宽度描述一个线程取多少字节，coalescing 描述同一 Warp 指令里各线程地址如何合并。当前方案每线程取 16 Bytes，线程之间又连续，因此两者同时成立；如果线程之间地址分散，即使每线程都是 int4，也可能产生很多低利用率的请求。

---

### 22. `int4` 加载为什么不会发生未对齐访问？

> 当前地址对齐可以从分配和偏移两方面证明。B 由 `cudaMalloc` 分配，基地址满足 int4 的对齐要求；每份 B 是 128 Bytes，每个 group 的四份 B 共 512 Bytes，这两个步长都是 16 的整数倍。
>
> 对于 group g、Warp lane l，实际读取地址是 `B_base + g * 512 + l * 16`。只要基址 16-byte aligned，每个 lane 的地址就都对齐。这里也不会越过自己的 B：sub_lane 从 0 到 7，每个线程覆盖四个相邻 int，最后一个线程恰好读取 B[28] 到 B[31]。
>
> v7 不仅 Global 源地址满足条件，Shared 目标 `s_b4` 还显式使用了 16 字节对齐声明。强制转换本身不会把未对齐地址变成对齐地址，因此真实系统接入时，必须把基址、步长和有效长度作为接口条件检查。

**可能追问：如果调用方传入的是偏移一个 int 的子指针怎么办？**

> 这种指针相对原地址偏移 4 Bytes，一般不再满足 16 字节对齐，不能直接沿用 int4 或 16-byte cp.async。可以选择标量或较窄加载的 fallback；如果要处理首尾后继续向量化，还需要同时保证读取范围、线程映射和 Shared 目标对齐，不能只改一个类型转换。

---

### 23. 这个优化是否偷偷改变了输入布局？

> 当前主线没有改变输入布局。v0 到 v9 的 B 都按 `[group][B][element]` 存储，v6/v7 改的是线程如何分工读取这些已有数据：以前是一个 Warp 每轮处理一份 B，后面是四个 8-lane subwarp 各处理一份 B，每线程拿四个连续元素。
>
> 这里要特别说明，A 被四份 B 复用并不是 v6 才有。v1 已经在一个 Warp 内把 A 排序一次，再顺序处理 B0 到 B3；v6 进一步把这 128 个 B 元素重新分配到 32 个线程上，使连续 int4 读取与本地四元素处理相匹配。
>
> 压缩包里的 `archive/transposed-layout/` 是另一组历史实验，使用 `[group][lane][B]`。归档 README 明确写了输入直接按该布局生成，没有计入转换成本。所以主线结果和历史布局实验必须分别介绍，不能混成同一个 v6/v7。

**可能追问：既然改布局可能更方便向量化，为什么要保留 B-major？**

> 因为上游和下游通常已经约定好数据格式。主线保持 B-major，能让版本比较围绕同一输入接口进行；如果采用新布局，就要把转换、上游写入方式和下游消费一起评估。仅仅在计时前免费重排输入，不能证明真实系统会更快。

---

### 24. 一个 Warp 读取四份 B，会不会产生四次内存事务？

> 不能直接把“四份 B”理解成“四次 DRAM 事务”。在当前映射中，一条 Warp 级宽加载涉及 32 个 lane，每个 lane 请求 16 Bytes，共覆盖连续 512 Bytes。硬件还会按具体层级把它拆成 request、cache line 或 sector，数这些单位时必须先说明层级。
>
> 如果只从 32-byte sector 的地址覆盖看，对齐的 512 Bytes 涉及 16 个 sector；如果按 128-byte 区间看，则覆盖四个这样的区间。这不等于可以直接断言某个 NCU request 指标必然是 4，因为宽指令的拆分、缓存命中和所统计的层级都会影响结果。
>
> 当前设计的价值是这些请求覆盖连续且有效的数据。它改善的是单线程搬运宽度和地址组织，而不是把 512 Bytes 神奇地压成一个事务。实际效果应结合 v6 的宽加载 SASS 和 NCU 的 sector、字节及吞吐指标判断。

**可能追问：数据连续就一定完全没有访存浪费吗？**

> 也不能这样保证。还要看对齐、活跃线程和访问边界；而整个 Kernel 的统计还混有 A load、结果写回以及 Shared 操作。判断某条 B load 的质量，应定位到对应指令，避免拿全 Kernel 的平均 bytes/request 直接反推它一定不合并。

---

## cp.async 与 SASS

### 25. 为什么使用 `cp.async`？

> v7 使用 cp.async 的目标，是在保持 v6 的 4×8 映射和相同 B-major 输入下，尝试提前搬运 B，并让它与独立的 A 加载和排序重叠。源码中每个线程调用 `cp_async_16(&s_b4[tx], b_source)`，提交后处理 A，等待完成，再读取自己的 `s_b4[tx]`。
>
> 这条路径是 Global 到 Shared，再由 Shared 到寄存器。相比“本来就需要把 Global 数据放入 Shared”的普通 LDG 加 STS 实现，cp.async 可以省去显式中间寄存器搬运；但本项目的直接对照 v6 原本就是 Global 到寄存器，不需要先写 Shared。因此不能把 v7 描述成无条件删除了 v6 的一次 STS，实际上 v7 新增了 8192 Bytes 的 B staging 和后续 LDS。
>
> `commit_group` 负责划分并提交当前线程的异步 copy group，`wait_group 0` 等待此前已提交组完成。它们能表达异步搬运依赖，却不会减少 B 的逻辑输入字节，也不自动保证编译后的重叠窗口足够大。

**可能追问：这里是一条多级流水线吗？**

> 主线 v7 不是跨多个 group 的双缓冲流水线。每个 Warp 只处理一个 group，B 发射一次、等待一次，并尝试与该 group 的 A 工作重叠。归档中的两组流水实验是另一份代码，不能把它的机制写到主线 v7 身上。

---

### 26. 为什么 v7 的 `cp.async` 只快了约 0.05%？

> 先明确，“约 0.05%”是 README 对既有实验的描述，单凭这个量级不能证明稳定收益。参考表把 v6 和 v7 分别列为约 3.019 ms、3.018 ms，精度已经不足以重算很细的差异；采集前 result 文件里的两次 Event 时间又是 3.018097 ms 和 3.016868 ms，它们属于另一组记录。
>
> 从机制看，v6 本来就已经很接近 DRAM 上限，报告中 DRAM Throughput 为 95.04%，所以异步搬运很难再明显提高整体吞吐。同时，v7 的实际 LDGSTS 被放到排序接近结尾的位置，搬运与独立工作的机器指令窗口较短。
>
> 因此我的结论是：v6/v7 在这个 workload 下性能非常接近，cp.async 验证了另一种数据路径，但不能承担主要加速成果。约 1.38× 的核心收益来自此前的排序加二分重构，细微差异需要交错测试和波动统计才能进一步确认。

**可能追问：如果测出来只差千分之一甚至更小，还值得保留吗？**

> 研究版本可以保留，因为它解释了编译器调度和异步路径；生产选择则要综合复杂度、兼容性和统计收益。若差异落在噪声内，直接 int4 的 v6 可能更省维护成本，不能为了使用 cp.async 就宣称它一定更优。

---

### 27. 为什么只看 CUDA 源码不能判断 `cp.async` 是否重叠？

> 因为 CUDA 源码表达的是程序依赖，最终指令仍会经过编译器调度。v7 源码先 issue B 的 cp.async，再加载和排序 A，但上传的 SASS 显示，A load 在偏移 0x00c0，B 的 `LDGSTS.E.BYPASS.128` 到 0x0670 才出现。
>
> 继续看附近指令：0x0680 是 LDGDEPBAR，0x06c0 是 DEPBAR，0x06e0 是 LDS.128，而最后一条排序 SHFL.BFLY 在 0x06f0。这说明 B 搬运确实发生在大部分排序之后，甚至最终排序指令还被穿插到了 B 读回之后。简单画成“完整排序，再等 B”也不够精确。
>
> 所以我会同时定位 issue、wait、consumer 和独立计算，确认可用的重叠窗口。SASS 能证明静态指令顺序和依赖结构，但不能单独测出实际隐藏了多少周期，真正的性能收益还要结合运行时间和计数器。

**可能追问：看到 LDGSTS 出现在计算前面，就能说延迟全部隐藏了吗？**

> 不能。它只说明存在重叠机会；还要看独立计算有多长、数据实际返回时间、资源竞争和 wait 是否仍会阻塞。v8 的 issue 确实更早，但增加的 Shared 操作和同步使总时间没有出现明显改善，这正说明顺序正确和性能受益是两件事。

---

### 28. v8 做了什么，为什么反而更慢？

> v8 是在 v7 基础上增加强制调度依赖的实验，主体仍在 `src/v6.cu`，由 `SUBWARP_FORCE_EARLY_CP_ASYNC` 宏选择。它先把 A 写进 volatile Shared 数组，再发射 B 的异步 copy，经过一次 block barrier 后读回 A 并完成排序。
>
> 源码里排序结束后还有一次 sorted A 的 Shared 写回与读回，用来把等待点约束在排序之后。因此它付出的不只是一次额外 barrier，还有排序前后两处 Shared round-trip。新增数组是 512 个 int，静态 Shared 从 17152 Bytes 增加到 19200 Bytes。
>
> 上传的 v8 SASS 可以核对到 LDGSTS 在 0x0190，排序从 0x01e0 开始，wait 在 0x07b0，确实给完整排序留出了重叠窗口。README 参考均值为约 3.021 ms，略慢于 v7；合理解释是扩大窗口没有抵消新增成本，但这么小的差异也需要统计验证，不能据此精确量化某条 barrier 的代价。

**可能追问：v8 中的 block barrier 本身会等待 cp.async 完成吗？**

> 不能把这次 barrier 当成异步 copy 的完成等待。代码仍然保留 cp.async.wait_group 0，再读取 B。这里 barrier 主要参与建立 A staging 的同步和调度结构；异步 copy 的完成条件要由对应的 wait 机制保证。

---

### 29. v9 为什么要强制 B 在 A 前面？结果怎样？

> v9 尝试通过真实的 device 函数调用边界控制顺序，避免 v8 把 A 暂存到 Shared 的办法。`src/v9.cu` 定义 `SUBWARP_FORCE_CP_BEFORE_A`，然后包含 v6 主体；第一个 noinline helper 发射并提交 B，返回后才加载和排序 A。
>
> 第二个 helper 接收并返回 sorted_a，内部执行异步等待，用这个数据依赖把等待留在排序之后。设计目标是“先 issue B，再 load/sort A，最后 wait 并消费 B”，同时不新增 v8 的 A staging 数组和额外 block barrier。
>
> 不过需要区分证据：压缩包有 v9 源码，README 记录了约 3.030 ms 的均值，并称 SASS 达到了预期；但包内没有 profiles/v9，因此我不能说已经从附件中的 v9 原始 SASS 独立确认了该结果。两个 noinline 调用带来的开销是合理分析，精确归因需要补齐该版本产物。

**可能追问：为什么不直接用 asm volatile 和 memory clobber 控制顺序？**

> 现有 cp.async helper 已经使用了这些写法，但 v7 的最终指令仍被下沉。它们不能保证任意无依赖的寄存器计算都保持源码顺序。v9 是用更强的调用和数据依赖做调度实验，是否最终达到目标仍要检查对应编译产物。

---

### 30. 这些失败版本为什么还保留？

> 这些版本保留的是优化假设和验证过程，不只是一个速度排名。v1 说明排序加二分能明显改善原来的执行路径；v1.5 尝试 register gather compact，参考时间反而变成约 3.147 ms；v2、v3、v4 又分别测试 ILP 和固定搜索结构，结果基本停留在同一性能区间。
>
> v6 到 v8 则回答了另一个问题：直接向量读取已经很快时，异步搬运和扩大调度窗口还能带来多少收益。v7、v8 的源码和 SASS 都能看到机制差异，最终时间却没有出现显著突破，这对判断优化空间很有价值。
>
> 我会保留实验代码、条件和结论，但把“已经有附件证据”和“只有 README 记录”的内容分开。尤其 v9 缺原始 profile，不能把保留了一个源文件等同于已经具备完整复现证据。

**可能追问：v1.5 的 register gather 为什么可能比 Shared scatter 更慢？**

> 它要根据 valid_mask 找第 k 个有效源 lane，再执行 Shuffle gather，额外引入了位选择和交换工作；而原来的一个 Warp 对一份 B 的 rank scatter，本身并不因为 rank 动态就必然有 bank conflict。它改变了组织方式，却未必减少真正的瓶颈，因此需要看生成指令和时间，而不是认为寄存器方案天然更快。

---

## Profiling 与性能判断

### 31. 你主要看了哪些 Nsight Compute 指标？

> 我会先看整体资源压力，再用具体 stall 和指令解释原因。上传的 v0 details 中，DRAM 是 56.19%，L1/TEX 约 97.90%，Mem Pipes Busy 为 97.85%；Warp State 还明确报告 MIO instruction queue 等待约 28.3 cycles，占平均发射间隔约 74.1%。因此不能只看到 Compute 97.85%，就说整数 ALU 或浮点算力已经跑满。
>
> v7 的 DRAM 达到 94.75%，Memory Workload 中的带宽为 930.73 GB/s，No Eligible 从 v0 的 72.14% 降为 43.92%。结合算法改变和多个后续版本时间收敛，更准确的项目描述是：主要压力从搜索相关的指令供给、MIO/L1TEX 路径，转向了 DRAM 带宽。
>
> 我还会看 registers、Shared、occupancy 和 Source Counters，避免只围绕单个百分比下结论。以下是附件 details 中的实际对照，Duration 不与 result 文件的 Event 时间混用。
>
> | 指标 | v0 details | v7 details |
> | --- | ---: | ---: |
> | NCU Duration | 5.10 ms | 3.02 ms |
> | DRAM Throughput | 56.19% | 94.75% |
> | Compute (SM) Throughput | 97.85% | 76.92% |
> | No Eligible | 72.14% | 43.92% |
> | Eligible Warps/Scheduler | 1.43 | 2.48 |
> | Registers/thread | 30 | 27 |
> | Achieved Occupancy | 88.63% | 96.36% |

**可能追问：MIO Throttle 和 LG Throttle 能直接当成同一个原因吗？**

> 不能。它们对应不同的指令队列节流分类，不能因为都和内存有关就互换。对这个附件，我能直接指出的是 v0 details 明确写了 MIO 队列等待；要进一步定位是哪条指令贡献最多，应对照 stall sampling 和源码/SASS。指标定义可参照 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)。

---

### 32. 你说的 934 GB/s 和 94.9% 是同一个指标吗？

> 不是同一个指标。程序里的 Effective BW 是逻辑字节数除以 CUDA Event 时间。当前每个 group 读 A 128 Bytes，读四份 B 共 512 Bytes，再输出四个 count 和四个 checksum 共 32 Bytes，因此每组是 672 Bytes，总逻辑流量为 2818572288 Bytes，约 2.819 GB。
>
> 而 v7 details 的 930.73 GB/s 和 DRAM Throughput 94.75% 来自 NCU 的硬件统计。前者用于描述测得的带宽，后者是相对工具所用峰值口径的比例；缓存、请求组织和测量环境都可能让它与程序推算值不同。
>
> 还要把 94.9% 的历史来源讲清楚：项目 README 将该数字关联到早期 lane-major 实验；主线 v7 的附件报告是 94.75%。面试可以概括为“主线已接近 95%”，但展示具体数字时应引用具体版本的报告。

**可能追问：为什么 930.73 GB/s 除以规格中的 1008 GB/s 不是 94.75%？**

> 因为分母不同。1008 GB/s 是按显存数据率和总线宽度推算的规格值，NCU 的百分比采用其对应的峰值和时钟口径。约 92.3% 和 94.75% 不能在不说明分母的情况下互换，更不能拿 logical bandwidth 算出的比例冒充 NCU 的 DRAM 指标。

---

### 33. 为什么 NCU 中的 Kernel 时间可能和 benchmark 不完全一致？

> 这个项目有三种容易混在一起的时间。第一是 README 声明的五轮参考均值；第二是 profiles 目录下 result.txt 记录的、采集 NCU 前单独运行程序得到的 Event 平均时间；第三才是 ncu_details.txt 里的 NCU Duration。
>
> 例如 v0 的 README 为 4.208 ms，result 文件为 4.477030 ms，details 里的 Duration 为 5.10 ms。v7 对应约 3.018 ms、3.016868 ms 和 3.02 ms。可见 result 文件放在 profiles 目录下，并不代表里面的时间就是 profiler 的 Duration。
>
> NCU 为收集指标可能执行多 pass，缓存与时钟控制也可能改变环境；不同批次正常运行还会受到温度和后台负载影响。因此正常 benchmark 用于报告加速比，同一份 profile 用于解释硬件行为，两者不能各取一个有利数字拼成结果。

**可能追问：那 4.48 ms 能不能写成这次 NCU 测得的 baseline？**

> 不能。附件中 4.477030 ms 来自 v0_result 的 CUDA Event 测量，v0 NCU Duration 是 5.10 ms。回答时直接给出文件来源即可；如果要比较正式加速比，则重新统一 baseline、运行条件和统计方式，而不是仅仅把文件夹名字当作时间来源。

---

### 34. 1.38x 是怎样计算的？

> 加速比首先要固定 baseline。简历中的约 1.38×对应 shared-broadcast brute-force 到推荐版本，而不是寄存器基线和优化版随意组合。按 README 的参考均值计算，`4.154 / 3.018 ≈ 1.376`，所以可写为约 1.38×。
>
> 同一组数的耗时降幅是 `(4.154 - 3.018) / 4.154 ≈ 27.35%`。加速比提高约 38%与耗时降低约 27%是不同表达，不能把耗时降幅也写成 38%。若选寄存器基线 4.208 ms，则得到约 1.39×，应明确这是另一个分母。
>
> 还有一个小细节：4.154 ms 按通常规则保留两位小数是 4.15 ms，不是 4.16 ms。因此“4.16→3.02”应作为原简历近似口径或另一批实验记录说明，不能说它是当前 README 数字直接四舍五入得到的。最稳妥的讲法是保留 4.154 和 3.018 的原始表值，再给出约 1.38×。

**可能追问：能不能用这次 result 文件里的 4.463923 ms 重新计算更高的加速比？**

> 可以把它作为另一批实验的观察，但必须与同批、同条件的优化结果完整报告，不能替换原来参考表的分母来美化简历。正式结论应记录 baseline、重复次数和运行环境，并保持一组统计口径一致。

---

### 35. 为什么 v1 到 v9 的时间几乎一样？

> 准确地说，v1 之后多数版本集中在约 3.02～3.04 ms，但不是所有版本都一样：v1.5 的参考均值约 3.147 ms，是明显的负向实验。v1 已经完成 A 排序一次和 B 二分搜索，主要瓶颈很快接近 DRAM 带宽。
>
> v2/v4 尝试增加 ILP，v3/v4 简化搜索状态，v5 合并输出，v6 改映射和宽加载，v7/v8 改异步搬运及调度。它们会改变指令结构、Shared 流量和 Warp 状态，但默认输入仍需处理每组 672 Bytes 的逻辑 Global 流量。
>
> 所以当 DRAM 已经接近上限时，局部改善难以明显缩短整段执行时间。反过来，新增开销足够大时仍会变慢，因此“已经 bandwidth bound”不等于任意增加计算都免费，也不能把所有失败版本都只用带宽一句话解释掉。

**可能追问：这是不是说明后续优化完全没有意义？**

> 后续版本帮助确认哪些局部指标能转化成收益，也提供不同接口和架构下的候选方案。但在当前条件下，应把它们定位为小幅调优和机制验证；如果想取得更大收益，需要优先考虑减少字节数、改变实际消费路径或消除系统边界开销。

---

### 36. Occupancy 是不是越高越好？

> Occupancy 反映驻留活跃 Warp 数相对硬件上限的比例，它提供隐藏延迟的机会，但不是执行效率或实际带宽本身。Warp 驻留着仍可能等数据、等队列或等同步，所以不能只追求一个接近 100% 的数字。
>
> 当前 v7 的理论 occupancy 是 100%，achieved occupancy 是 96.36%，寄存器为 27/thread。与 v0 相比，它确实提供了足够的驻留工作，但主要优化机制仍是搜索和数据路径改变，而不是简单把 occupancy 从 88.63% 拉到 96.36%。
>
> 附件还有一个很直观的反例：v3 的 achieved occupancy 约 71.45%，DRAM 却仍达 94.45%，正常时间也约 3.038 ms。这说明在当前流式 workload 中，不同调度状态可能已经足够支撑接近峰值的带宽，不能按 occupancy 排名直接给 Kernel 排名。

**可能追问：要不要用 maxrregcount 强行把寄存器压低？**

> 只有发现寄存器确实限制目标驻留数时，才值得作为对照实验。强压寄存器可能增加 spill 或重复计算。当前 v7 已达到理论 48 warps/SM 的上限，继续降低寄存器不会凭空增加更多驻留 Warp，应同时看 spill、指令和时间。

---

### 37. `No Eligible` 仍有约 44%，为什么还说是带宽瓶颈？

> No Eligible 约 43.92%表示在一部分 scheduler 周期中，没有 ready-to-issue 的 Warp。它描述的是发射状态，不能直接解释成“GPU 有 44%的时间没做事”，也不能说明所有等待都来自同一种原因。
>
> v7 同时表现为 DRAM Throughput 94.75%、约 930.73 GB/s 的带宽，以及较高 achieved occupancy。内存请求发射后，DRAM 可以在 scheduler 暂时没有新指令时继续处理已有请求，所以发射空闲和显存繁忙可以同时存在。
>
> 我的判断依赖这些证据的组合：大规模流式输入、DRAM 接近上限、多种局部优化的时间接近。因此整体更符合带宽受限；但仅凭当前汇总表不能断言剩余 44%的空槽全部是 DRAM 等待，还需要具体的 stall sampling 来拆解。

**可能追问：把 No Eligible 从 44%降到 20%，一定会变快吗？**

> 不一定。v2 的 No Eligible 约 30.19%，低于 v7，但并没有比 v7 更快。这个项目已经给出实例：改善发射状态不必然改善最终吞吐，还要看新增指令、数据字节和当前最紧的硬件限制。

---

## 实验设计与严谨性

### 38. 你怎样保证各版本比较公平？

> 我会固定可比较的工作契约：默认 4194304 个 group，每组 A[32] 对四份 B[32]，统一 B-major 输入，输出 survivor count 和 checksum，默认 512 threads/block。初始化不计入时间，正常计时先 warmup 10 次，再用 CUDA Event 平均 50 次 launch。
>
> 构建入口统一使用 Makefile 的 sm_89 目标，benchmark.sh 默认再运行五轮，并解析 Wrong results，非零就停止。虽然一些早期程序即使发现校验错误也返回 0，脚本仍会检查输出内容，因此不能只看进程退出码就宣称所有版本正确。
>
> 公平不代表所有内部操作完全相同：v1.5 改 compact，v5 增加输出 staging，v6 重写映射；它们保持的是输入和可观察输出契约。当前脚本按版本顺序连续运行，也没有自动完成锁频、随机化顺序和方差报告，这些应作为下一步测量完善项，不应写成已经实施。

**可能追问：只修改 NUM_B_LISTS 后执行 make all 就能公平比较任意 N 吗？**

> 不能。v6～v9 有 NUM_B_LISTS==4 的静态断言，不能直接用于其他 N；而且 Make 不会因为命令行变量变化就必然重编译已有目标。改变参数时应选择支持该参数的版本，并使用独立构建目录或重新构建，确认实际二进制的配置。

---

### 39. 正确性是怎样验证的？

> 当前正确性检查基于确定性输入。A 是 0、2、4……62 的一个排列，具体由 `((lane * 17) & 31) * 2` 生成；B 的偶数元素位置选取 A 中已有值，奇数位置生成 `4096 + b_id * 128 + lane`，这些值都不在 A 中。
>
> 因此每份 B 应保留 16 个元素，checksum 可以直接算成 `16 * 4096 + 16 * b_id * 128 + 256`，其中 256 是 1、3……31 的和。Host 会检查全部 group 和四份 B 的 count 与 checksum，现有附件结果都报告 Wrong results: 0。
>
> 这个检查能确认当前构造输入下的输出一致性，但覆盖范围有限：所有 group 的数值模式重复，miss 都大于 A 的最大值，而且 sum 无法检测顺序错误或所有碰撞。更完整的验证需要随机输入、区间内 miss、重复 key、极值和逐元素 reference，这些不能说成现有代码已经全部做过。

**可能追问：checksum 正确，为什么仍不能证明 compact 顺序正确？**

> 因为交换两个 survivor 不会改变求和结果；甚至有些数值错误也可能相互抵消。顺序正确性需要依靠 rank 的逻辑证明，并用完整列表与 CPU 稳定过滤结果逐元素比较。count 加 checksum 是回归检查手段，不是完整正确性证明。

---

### 40. 为什么输出 checksum，而不是完整候选数组？

> 当前 microbenchmark 希望让过滤后的数据继续在同一个 Kernel 内被消费，所以把 survivor compact 到 Shared Memory，再读出求 checksum，最后仅写每份 B 的 count 和 checksum。它模拟了设备端后续消费的一段路径，避免把大规模完整结果写回作为当前实验的主要工作。
>
> 默认一份 B 保留 16 个 int，若全部写回，单份 B 还会增加 64 Bytes payload，全局总输出流量就明显不同。当前程序用于计算 Effective BW 的每组 672 Bytes，只包含 A、B 和 count/checksum，不包含完整 compact payload。
>
> 因此我会明确说，3.02 ms 对应当前过滤、Shared compact 和校验消费接口。它不能直接代表“完整候选数组已经输出到 Global”的耗时，更不代表上层图构建或检索系统的总耗时。真实接入需要按实际消费者重新定义接口和 baseline。

**可能追问：如果业务真的只需要 count 和 sum，是不是可以不做 compact？**

> 可以另写仅统计的算法，直接对有效 B 值规约，省掉形成紧凑列表的步骤。但那改变了这个 benchmark 想保留的内部消费路径，不能拿少做 compact 的时间声称同一过滤接口被优化了。应把 count-only 或 count-sum 作为单独任务比较。

---

### 41. 测试数据会不会过于理想化？

> 会。当前输入是为了稳定比较 Kernel 而构造的：A 是固定偶数集合的排列，B 固定一半命中、一半未命中，而且未命中值全部大于 A 的最大值。所有 group 的数值模式也相同，只是存放在不同地址。
>
> 这使每份 B 的 survivor 数固定为 16，compact 工作量和分支模式比较规律，适合隔离版本差异，但不能代表任意真实候选分布。尤其二分的 miss 只覆盖一个方向，并没有充分验证小于最小值、落在两个 A 值之间等情况。
>
> 因此性能结论应限定为固定 32、1:4、当前数据模式和 RTX 4090。工程化时我会分别改变重复率与有效位置分布，再加入随机 A/B、边界 key 和真实样本，观察搜索、Shared conflict 与带宽是否变化，而不是承诺任何输入都能获得相同加速比。

**可能追问：所有 group 的值相同，会不会就不需要读几 GiB 数据了？**

> 不会自动消除读取，因为它们仍是不同地址，Kernel 也没有通过公式替代输入。附件 v7 的 L2 Compression Success Rate 为 0、L2 Hit Rate 约 4.79%，支持实际发生了大规模数据访问。但相同数值模式会让控制行为更规律，所以仍要补不同数据分布。

---

### 42. 为什么 RTX 3090 上版本差距可能不同？

> 版本排名取决于搜索指令供给、显存带宽、缓存和资源占用之间的平衡，换 GPU 后这个平衡可能变化。4090 上 v1 已经接近带宽上限，因此后面的 ILP、宽加载和异步调度很难拉开差距；3090 上不能预先假设这些条件完全相同。
>
> 我会先保持 A/B 布局、group 数和输出语义一致，针对目标架构重新编译，再比较正常时间、DRAM、MIO/scoreboard 等 stall、寄存器与 occupancy。既要检查算法是否仍正确，也要检查 int4 和 cp.async 生成了什么 SASS。
>
> 当前压缩包没有一套对应的 3090 结果，所以只能解释迁移方法，不能给出已经测得的 3090 加速比。也不能仅按两张卡理论带宽的比例换算实际时间，因为执行和调度路径也会参与限制。

**可能追问：当前 Makefile 直接就能切换到 sm_86 吗？**

> 它的编译规则写死了 -arch=sm_89，没有独立的架构变量。迁移时需要调整构建目标或使用明确的目标编译命令，并放进单独的构建目录，避免误用原 sm_89 二进制。源码相同不代表可以忽略目标架构和编译器版本。

---

### 43. 如何避免 GPU 温度和动态频率影响结论？

> 先区分已经实现的计时措施和还需要补充的实验控制。现有程序有 10 次 warmup、50 次 launch 平均，benchmark.sh 默认运行五轮；但脚本是按版本顺序连续执行，并没有自动随机化、监控温度或输出标准差。
>
> 对于约 1.38× 的主要收益，现有参考数据提供了明显差异；对于 v6/v7 的极小差别，更容易受到频率、功耗、温度和后台负载影响。我会采用交错顺序、多轮重复，并记录这些状态；设备支持时再使用适当的时钟控制。
>
> 报告中应同时给均值和波动范围。若差别落在自然波动内，就写“性能接近”，而不是因为某次小数点后更小就宣布稳定收益。SASS 可以说明机制改变，却不能代替统计证明。

**可能追问：warmup 之后还会发生热漂移吗？**

> 会。warmup 能减少首次初始化和初始状态差异，但不保证长时间测试中温度、boost 和功耗状态完全不变。因此不同版本最好交错执行，并保留逐轮原始时间，检查结果是否随测试顺序单调变化。

---

## 设计取舍与追问

### 44. 二路 ILP 为什么没有明显加速？

> v2 的核心是在一个线程内维护两套二分状态，每轮先发出两个独立的 Shuffle，再更新两套 left/right/existed，让一条搜索链等待结果时，有另一条链可供编译器调度。它增加的是线程内 ILP，不是增加 Warp 数量。
>
> 但同时维护状态也增加工作量和寄存器需求。附件中 v1 是 26 registers/thread，v2 是 29；v2 的 No Eligible 确实更低，但参考均值仅从约 3.031 ms 变成 3.030 ms，基本持平。此时 DRAM 已经很忙，调度改善不一定能转化成更高吞吐。
>
> 所以我不会说“ILP 无效”，而是说它改变了指令组织和发射状态，但当前 workload 的主要限制已经不在这一点上。是否值得使用，仍要看目标数据和 GPU 上的最终时间。

**可能追问：为什么不继续做四路或八路 ILP？**

> 搜索状态和 live value 会继续增加，可能造成更高寄存器压力、更多指令，甚至 spill。当前两路实验都没有明显收益，继续加宽应先有新的 stall 证据，而不是认为并行链越多必然越快。

---

### 45. 为什么固定 32 专用搜索也没有超过通用版本？

> v3 的 fixed-32 搜索从 pos=-1 开始，按 16、8、4、2、1 的步长探测，最后读取 pos+1 并判断相等。它主要减少通用二分的边界和命中状态更新；v4 在这个结构上又做两路 ILP。
>
> 这里必须看实际代码：五次定位之后还有一次最终 Shuffle，因此每个 target 仍有六次 pivot 读取。不能把“固定 32”直接宣传成从六次读取降为五次。v7 也没有采用这个函数，而是继续使用带 left/right/existed 的通用六轮搜索。
>
> 参考表中 v3/v4 都约 3.038 ms，与 v1 接近。它们没有减少 A/B 必须访问的字节，且各自会改变依赖链和调度，所以更少控制状态并不保证更低时间。

**可能追问：target 大于 A 最大值时，fixed-32 会不会访问 lane32？**

> 这份实现的五个步长之和是 31，pos 从 -1 开始最多到 30，最终 candidate 最大为 31。它会读取 A[31] 做相等判断并返回 false，所以不会访问 lane32；但这也说明它不是直接返回标准 lower_bound 插入位置的通用函数。

---

### 46. 下一步还可以怎样优化？

> 我会先补充适用范围，再决定是否继续微调。现有数据集中在 1:4、50%重复率和固定形状，下一步应测试随机命中位置、不同重复率、N 和 block size，明确当前推荐方案在哪些条件下成立。
>
> 局部优化可以围绕 compact Shared store 的 excessive wavefront 做地址映射对照，但不能只看到 conflict 就预判收益。当前 v7 已接近 95% DRAM，新的 swizzle 或分阶段写入可能增加地址计算、指令和同步，反而抵消改善。
>
> 更大的潜力来自真实调用方：如果后续 GPU 阶段可以直接消费 survivor，就研究融合和缓冲区复用，减少全局写回、重新读取或 launch。附件没有完整系统实现，因此这些是下一步方向，不是当前已经取得的 E2E 成果。

**可能追问：为什么不继续把二分代码再缩短几条指令？**

> 可以做对照，但预期收益有限，而且 v3/v4 已经探索过简化控制。当前更重要的是确认数据契约和总字节成本；当已有输入占满带宽时，仅压缩几条整数指令未必缩短关键路径。

---

### 47. 如果 N 不是 4，这个映射还能用吗？

> 当前主线 v6～v9 不能只改 N 就直接使用，因为公共主体有 `static_assert(NUM_B_LISTS == 4)`，并且 b_id=lane>>3、每线程四元素、B 指针和 Shared 分配都围绕四份 B 设计。
>
> 如果需要其他 N，可以先参考 v1 那种一个 Warp 顺序遍历多份 B 的结构，它保留一次 A 排序的复用；再针对 N 重新考虑 subwarp 和向量宽度。例如 N=2 可以让每份 B 使用 16 lanes、每线程两个元素，N=8 可以分两批处理。
>
> 改动后还要检查输出写回，而不只是 B 加载。尤其 RESULTS_PER_BLOCK 随 N 增大，若超过 block 线程数，原来每线程只写一个结果的方式就不够，需要步进循环覆盖输出。

**可能追问：算法能扩展，为什么还要写静态断言？**

> 因为它阻止调用方把专用映射当成通用接口误用。删除断言并不会自动修复地址、mask 和结果分配。更好的做法是保留当前 1:4 fast path，再为其他 N 提供经过验证的实现。

---

### 48. 如果候选长度不是 32 怎么办？

> 固定 32 是当前实现的重要前提：A 每 lane 一个值，排序网络固定 15 stage，pivot 可由一次 Warp Shuffle 取得，B 的有效位也能装进一个 32-bit mask。长度变化会影响这些对应关系。
>
> 小于 32 时可以保留完整 Warp，安全加载有效元素并给无效 A 加 padding，搜索只针对真实长度或携带 valid 信息；无效 B 必须强制判为不参与输出。不能让 padding 值与真实 key 混淆，也不能让参与 Shuffle 的线程随意退出。
>
> 大于 32 时可以每线程保存多个值、分块搜索，或转向 block 级 Shared 结构。并不是一定要多个 Warp，但原来的一次 Shuffle 就不能直接覆盖全部 A。应重新比较算法、资源和同步成本，不能沿用 3.02 ms 的结论。

**可能追问：长度是 24，能不能直接用一个 24-lane 的 Shuffle width？**

> 不能把任意长度直接作为当前 Shuffle 分区宽度。更简单的设计是保留 32-lane 执行，用合法的宽度和明确的 padding、长度判断处理 24 个有效元素。逻辑长度和硬件通信分区宽度要分开。

---

### 49. 你个人在这个项目中的核心贡献是什么？

> 我会把核心工作讲成一条可以沿着源码和实验解释的链：先明确固定 32、1:N membership filtering 的输入和输出，建立两种暴力 baseline；再通过 A 排序一次和 Shuffle 二分降低重复搜索成本。
>
> 随后围绕实际数据路径做版本对照：v5 合并输出，v6 将 B-major 的四份 B 映射成 4×8 subwarp 并使用 int4，v7/v8 探索异步搬运和调度窗口。同时用正确性、Event、NCU 和 SASS 判断哪些变化有收益、哪些只是机制变化。
>
> 个人贡献要对应自己实际负责的设计、代码审阅和实验分析，不能因为仓库里存在某段代码就自动把全部工作归为独立手写。这个项目最值得讲清楚的是能解释选择、定位问题和验证结论，而不是只罗列优化名词。

**可能追问：如果代码使用了 AI 或参考实现辅助，你怎么回答？**

> 我会如实说明辅助范围，并重点展示自己理解和验证过的内容，例如线程映射、同步条件、数据流量和性能归因。能复述工具生成的代码不等于掌握项目；我需要能够解释它为什么正确、怎样验证，以及哪些结果没有证据支持。

---

### 50. 这个项目最大的技术难点是什么？

> 我认为最大的难点是把源码设计、编译后的实际行为和最终性能对应起来。比如源码中提前写了 cp.async，并不能保证 SASS 中的 LDGSTS 足够早；即使 v8 把它放到完整排序前，也没有明显降低总时间。
>
> 另一个难点是区分指标含义。当前 v0 Compute Throughput 很高，但报告直接指出 MIO 队列等待，不能把它解释成 FP32 算力跑满；v7 的总执行指令数也并没有比 v0 更少，却明显更快，说明指令结构和瓶颈比总条数更重要。
>
> 因此这个项目的难点不只是实现二分，而是让每条性能判断都能回到对应的源码、测量条件和报告。把这些关系讲清楚，才能知道下一步该优化什么。

**可能追问：如果只允许讲一个具体难点案例，你选哪个？**

> 我会选 v7/v8 的异步调度对照：先说明源码想重叠 B 搬运与 A 排序，再展示两个版本的 issue/wait 位置，最后解释为什么扩大窗口仍没有明显收益。它同时涉及编译器、同步、资源成本和带宽约束。

---

## 硬件资源与底层追问

### 51. 你的 RTX 4090 有多少 Shared Memory？当前 Kernel 用了多少？

> 对 Ada 的资源上限，需要区分 SM 总容量和单 block 的可用容量。官方资料给出每 SM 最多 48 个驻留 Warp、64K 个 32-bit 寄存器，Shared 上限为 100 KiB；单 block opt-in 上限为 99 KiB，超过默认 48 KiB 的使用需要相应动态 Shared 配置。[NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
>
> 当前 v7 的用户静态 Shared 可以直接从源码算出来：compact 为 16×4×33×4=8448 Bytes，两个结果数组共 512 Bytes，B staging 为 512×16=8192 Bytes，总共 17152 Bytes。v8 再加 2048 Bytes 的 A staging，总计 19200 Bytes。
>
> 更关键的是实际配置：v7 的 NCU Shared Memory Configuration Size 为约 65.54 KB，即 64 KiB carveout，并不是每次都配置成设备最大 100 KiB。报告还列出了 driver Shared 资源，所以不能只拿最大容量除以用户数组大小来决定实际驻留数。

**可能追问：为什么报告里 17.15 KB 与手算 17152 Bytes 对得上？**

> 因为该表的 Kbyte 使用十进制显示，17152 Bytes 约为 17.152 KB；若换成 KiB，则是 16.75 KiB。讲资源时应明确单位，避免把报告显示值和以 1024 为单位的容量直接混算。

---

### 52. Shared Memory 超了会发生什么？

> Shared 超限有两类后果。若单 block 用量仍合法，但总资源不足以容纳原来的多个 block，就会减少同时驻留的 block，可能降低隐藏延迟的能力；若超过单 block 的允许容量，则不能按该配置执行。
>
> 当前 v7 的静态 Shared 只有 17152 Bytes，远低于默认 48 KiB。若未来增加到更大容量，超过默认限制的设计不能只扩大静态数组，需要按设备要求使用动态 Shared 和 opt-in 属性，同时保证用户请求、静态部分及驱动保留资源都满足条件。
>
> 静态用量不合法可能在编译阶段暴露，动态请求过大则可能在 launch 时失败。Shared 不会透明地溢出到 Global 来保证程序继续执行，因此资源变化必须结合编译输出、launch 错误检查和 occupancy 重新确认。

**可能追问：Shared 用量增加一点，occupancy 就会按比例降低吗？**

> 不是连续线性变化，而是按可驻留 block 数出现台阶。只要仍能放下三个 block，理论 Warp 数可能不变；一旦跨过阈值只能放两个，驻留数才明显下降。还要考虑资源分配粒度和实际 carveout。

---

### 53. Shared Memory 会像 CPU cache 一样自动换出吗？

> Shared Memory 是程序显式管理的片上存储，不是像 cache 一样由硬件按地址命中自动装入和逐出。一个 block 驻留时需要拥有自己的 Shared 资源，其他 block 不能通过普通指针把它当作透明后备内存使用。
>
> 因此当前 v7 的 compact、B staging 和结果暂存都要在 block 资源预算内。容量不足时，通常影响可驻留 block 数；若单 block 本身超限，就无法合法启动，不会自动把 s_compact 搬到 L2 或 DRAM。
>
> 这和寄存器 spill 的行为不同。开发者当然可以主动设计 Global scratch buffer 替代部分 Shared，但那是新的地址和同步方案，也会增加全局访存成本，不属于硬件自动换出。

**可能追问：Shared 和 L1 使用统一片上资源，为什么还说它不是 cache？**

> 物理资源可以共享，编程语义仍不同。L1 管理的是缓存行，Shared 则由程序显式寻址和同步；调整 carveout 会改变两者的容量分配，但不会让一个越界的 Shared 地址自动获得 Global 后备存储。

---

### 54. 当前 occupancy 为什么接近 100%，怎样手算？

> 默认每 block 有 512 threads，即 16 warps；该设备每 SM 最多 48 个驻留 Warp，所以线程维度允许三个 block，理论 occupancy 就是 `3×16/48=100%`。
>
> 然后要检查其他资源是否也允许三个 block。附件 v7 的 NCU 直接给出 Block Limit Registers=4、Block Limit Shared Mem=3、Block Limit Warps=3、Block Limit SM=24，取最小值为 3。因此实际报告中 Shared 和 Warp 限制都为 3，不能说 Shared 一定能放五个、完全不构成限制。
>
> 其中 Shared 按约 64 KiB 的实际 carveout 分配，除了 17152 Bytes 用户数组，还存在驱动保留和粒度约束。理论值描述可驻留上限，achieved occupancy 96.36%是运行时测得的活跃情况，两者不必完全相等。

**可能追问：用 27×512 计算寄存器，为什么只能作估算？**

> 因为寄存器资源按硬件规定的粒度分配，并非任意数量逐个精确塞满。粗算 65536/(27×512) 可得四个 block 的数量级，但最终应以编译资源、occupancy API 或 NCU 的资源限制表为准。

---

### 55. 寄存器使用过多会怎样？和 Shared Memory 超限相同吗？

> 寄存器用量首先影响一个 SM 能同时容纳多少线程和 block；当 live value 无法在分配到的寄存器中保存时，编译器还可能生成 spill，使用线程私有的 local memory。Local 是逻辑地址空间，不代表它是一块额外的片上寄存器。
>
> 这种额外存取会经过设备内存层次，增加指令和潜在流量，因此高 occupancy 不一定能弥补 spill 代价。Shared 则没有同样的透明 spill 机制，资源不足时是驻留减少或配置不合法。
>
> 当前 v7 报告是 27 registers/thread，寄存器允许的 block 数为 4，高于实际驻留的 3，所以它不是当前第一资源限制。若要严谨地说“没有 spill”，还应检查 ptxas 的 spill 统计或对应 local-memory 指标，不能只从寄存器数量推断。

**可能追问：寄存器很多但没有 spill，就一定没有问题吗？**

> 也不一定。即使没有 spill，较高用量仍可能降低驻留数；反过来，为降低用量而增加重复计算也可能变慢。需要同时比较资源、依赖链、occupancy 和运行时间。

---

### 56. Shared Memory bank conflict 到底怎样发生？broadcast 算冲突吗？

> 分析 bank conflict，要看同一条 Shared 指令在一次服务中的地址映射。对当前 int32 访问，常用 `bank=(byte_address/4)%32` 判断。如果多个 lane 访问同一 bank 的不同 word，可能需要额外 wavefront；多个线程读取完全相同的 word，则可利用 broadcast。
>
> v0_shared 每轮都读取同一个 A 元素，属于广播式供数。v6/v7 compact 则是四个 subwarp 写四个独立列表，rank 虽然在每份列表内唯一，却可能跨列表映射到相同 bank，所以“不发生写覆盖”和“没有 bank conflict”是不同结论。
>
> v7 details 报告平均约 1.6-way shared store conflict，Source Counters 又记录 8388608 excessive wavefront。两处统计范围不同，不能把所有 wavefront 都当冲突；宽访问本来需要的正常拆分也不能算额外冲突。

**可能追问：stride=33 是否已经解决了所有 bank conflict？**

> 没有。它让相邻列表的起始 bank 错开一个位置，但 rank 由有效元素分布决定，不同列表仍可能撞 bank。实际是否更好要比较对应 STS 的地址和计数器，不能只凭数组第二维加一就宣称无冲突。

---

### 57. `__syncwarp` 和 `__syncthreads` 分别在这里保证什么？

> 这份源码里两种同步服务于不同的通信范围。v7 在 compact store 后执行 `__syncwarp(FULL_MASK)`，确保同一 Warp 的 Shared 写入完成并可由其后续消费代码按顺序读取。
>
> 随后四个 subwarp leader 把 count/checksum 写入 block 级结果数组，而最后实际写 Global 的是 tx<64 的连续线程，生产者和消费者可能来自不同 Warp，因此这里需要 `__syncthreads()`。v8 还额外增加一次 block barrier，用于 A staging 的调度实验。
>
> v0/v1 的 compact buffer 会在处理下一份 B 时复用，所以有消费前、消费后的 Warp 同步；v7 每份 B 有独立 compact 区域，不能把前面版本的同步次数直接套过来。同步选择应依据谁写、谁读、何时复用。

**可能追问：Shuffle 的 sync 后缀可以替代 Shared 的同步吗？**

> 不能把它当成通用内存 barrier。Shuffle 用于对应参与线程之间的数据交换，而 Shared 的生产者、消费者需要合适的内存顺序。当前代码显式保留 syncwarp 和 syncthreads，正是为了表达这些数据依赖。

---

### 58. 为什么 Warp Shuffle 比 Shared Memory 适合保存 sorted A？

> A 只有 32 个 int，每个 lane 持有一个，刚好适合分布在 Warp 寄存器中。排序时通过 shfl_xor 取得 partner；搜索时每个 lane 通过动态 mid 的 shfl 取得 pivot，所以不需要建立 sorted A 的 Shared 数组。
>
> 这减少了 Shared 地址计算、写入和读取路径，同时让同一个 sorted A 被后续四份 B 复用。需要注意“分布在 Warp 寄存器中”不是说一个线程拥有可随机下标访问的 32 元素寄存器数组，而是每个线程只持有自己的标量，跨 lane 读取通过 Shuffle 实现。
>
> Shuffle 本身仍消耗指令和执行资源，连续搜索也存在依赖。因此本项目的收益不是“寄存器永远比 Shared 快”，而是采用了适合固定规模的算法与通信方式，并通过 baseline 和报告判断代价。

**可能追问：二分每个 lane 的 mid 不同，能用同一条 Shuffle 吗？**

> 可以。每个参与 lane 可以提供不同的 source lane，分别得到自己的 pivot。但参与范围仍要覆盖整个 A 的 32 个 lane；B 虽按 8-lane subwarp 分组，A 搜索不能错误地限制在本 subwarp 内。

---

### 59. 为什么二分时四个本地 B 不能只做 5 次比较？

> 当前 v6/v7 每线程持有四个 B 值，源码用 item=0 到 3 分别调用 warp_binary_search。每个值有自己的搜索路径，一次标量 Shuffle 对一个 lane 只能返回一个 pivot，不能同时给这个线程的四个 target 返回四个不同中点。
>
> 所以当前实现对应四套六轮搜索，Warp 层面约 24 次搜索 Shuffle，加上 A 排序的 15 个 stage，共 39 个 sort/search stage。这个数只是描述主要通信步骤，不等于全部 SASS 指令条数，也不等于每个 group 只做 39 次元素比较。
>
> 四个 subwarp 使同一条指令可以服务四份 B 的对应 item，但并没有消除一个线程处理四个 item 的工作。若交错四条搜索链，可以探索 ILP，但仍需付出相应的 pivot 读取和状态维护。

**可能追问：通用二分的六轮能给一个确切例子吗？**

> 可以。对排序后的 0、2……62，搜索最小值 0 时，中点依次是 16、8、4、2、1、0，共六次。当前函数每轮都判断相等，并不是先做固定五次完整查找再统一确认；命中的 lane 仍参与余下 Shuffle，只停止更新搜索状态。

---

## 场景题与设计变体

### 60. 如果是 1A-2B，v6/v7 的 4x8 映射还能直接用吗？

> 当前 v6/v7 直接编译 N=2 会被静态断言拒绝，因此需要重新设计映射。最容易验证的方案是保留一个 Warp 一组 A，用完整 32 lanes 排序，然后顺序处理两份 B，类似 v1 的结构。
>
> 如果希望一轮覆盖两个 B，可以让每份 B 分配 16 lanes、每 lane 用 int2 处理两个连续元素，形成 2×16 映射。这样同样使用一次 A 排序，但 mask 组合、元素位置、Shared 区域和结果 leader 都需要调整。
>
> 我不会优先把一个 Warp 切成两个独立 16-lane group，因为每个 group 的 A 仍有 32 个元素，需要每线程保存多个 A 并改排序和搜索。那是可研究的方案，但复杂度高于只改变 B 映射。

**可能追问：使用 2×16 时，二分 Shuffle 的 width 应该是 16 吗？**

> 不是。B 的分组宽度可以是 16，但 sorted A 仍分布在整个 Warp 的 32 个 lane，搜索 pivot 必须能访问全 Warp。只有 B 的 mask 合并或 checksum reduction 才适合限定在对应 16-lane 分区。

---

### 61. 如果是 1A-1B、1A-8B，应该怎样映射？

> 1A-1B 最直接的方案是一个 Warp 每 lane 一个 B，先加载 A，再比较暴力搜索和排序加二分的整体代价。一次 A 排序只能服务一份 B，摊销程度较低，但这不等于排序一定不划算，仍要看实际指令和计时。当前附件没有可独立核验的 1:1 对照结果，所以我不会说已经证明两者持平。
>
> 1A-8B 可以保留 sorted A，分两批处理四份 B。沿用 4×8 映射时，需要修改 B 地址中的 group stride、批次偏移、输出索引和 Shared buffer 复用同步，不能只删除 N==4 的断言。
>
> N 增大能进一步摊销 A 排序，但 B 字节和 compact 工作量也增加。评价时应看每份 B 的吞吐、资源和总时间，而不能只比较一个 group 的延迟。

**可能追问：N 越大，加速比就一定越大吗？**

> 不一定。算法复用更充分，但输入流量也更大，可能更早遇到带宽上限；若展开过多，还会增加代码和寄存器压力。应保持真实 N 和相同输出接口，比较支持该 N 的实现。

---

### 62. 如果列表长度小于 32 或不是 2 的幂怎么办？

> 如果只是处理少于 32 个有效元素，我倾向于先保留完整 Warp 的规则执行，再通过长度和有效位控制输入与输出。无效 A 位置可安全填充，但必须保证搜索不会把 padding 当成真实命中；无效 B 则不能进入 valid_mask 或 checksum。
>
> 例如 A 长度为 24，仍可运行 32 元素排序网络，把无效项放到有序末尾，并让搜索只覆盖前 24 个有效位置。也可以携带 valid bit 排序，支持完整 int key 域，避免用 INT_MAX 充当不存在的哨兵。
>
> 边界处理不能破坏 Warp collective 的参与条件。即使某线程没有真实输入，只要仍承担 padding A 或参与 full-mask Shuffle，就必须执行相应操作，同时避免越界 load。

**可能追问：只把 valid_mask 的高位清零就够了吗？**

> 不够。mask 清零只能控制后续输出，不能修复之前已经发生的越界读取，也不能保证 Shuffle 的源 lane 有合法数据。输入加载、排序、搜索和输出四个阶段都要有一致的有效性规则。

---

### 63. 如果 A/B 长度大于 32，为什么不能简单循环？

> 可以循环处理更长列表，但原实现的一些优势会消失。当前 A 的全部数据由 32 lanes 各持有一个标量，一条动态 Shuffle 就能选中任意 pivot；如果 A 有 64 个值，每 lane 至少要持有两个，搜索还需要确定取哪个寄存器分量。
>
> 排序也不再是现成的 15-stage 网络。可以采用每线程多元素排序、分块 A、Shared 二分或 hash；如果由多个 Warp 协作，还要增加跨 Warp 同步。B 更长相对容易分批，但 valid mask、输出 count 和 compact buffer 也要扩展。
>
> 所以“加一个循环”可能保持功能思路，却不一定保持原来的资源占用和访存效率。长度变化后应重新建立 baseline，把数据复用、顺序要求和真实输出成本一起比较。

**可能追问：A 大于 32 就一定需要多个 Warp 吗？**

> 不一定，一个 Warp 可以每线程持有多个元素。问题是单次 Shuffle 只能返回一个源值，额外寄存器选择、排序网络和资源成本必须处理。多 Warp 是候选设计之一，不是唯一合法实现。

---

### 64. 如果 B 指针没有 16-byte 对齐，你怎么处理？

> 先在接口上区分对齐 fast path 和安全 fallback。当前 int4 和 cp.async 16B 都依赖源地址满足 16-byte 对齐，v7 的 Shared 目标也必须满足同样要求；任意 slice pointer 不能因为来自 cudaMalloc 就自动符合条件。
>
> 若基址未对齐，最简单的正确做法是使用标量读取，把四个 int 分别装入线程本地值，再沿用搜索和 compact 逻辑。也可针对 8-byte 对齐提供较窄路径，但需要单独检查生成指令及有效范围。
>
> 只有在前后缀、有效字节和线程映射都清楚时，才考虑部分标量加中段向量化。当前每份 B 只有 128 Bytes，复杂对齐处理的成本可能不小，不能预先保证比统一标量 fallback 更快。

**可能追问：先向下取整到 16-byte 地址，再多读一点可以吗？**

> 不能默认这么做，因为可能读到分配范围之外，甚至跨越不可访问页。只有调用方明确提供合法的额外可读范围时才能设计这种路径；通用接口应保证每一次实际加载都在有效地址范围内。

---

### 65. 如果 B 中自己也有重复元素，当前算法会去掉吗？

> 不会。当前 duplicated 的定义是“这个 B 元素是否存在于 A”，不是“这个值是否在 B 中出现过”。每个 target 独立查询 sorted A，未命中的位置都作为 survivor，随后按 B 原顺序 compact。
>
> 例如 A 不含 7，而 B 有两个 7，这两个位置都会保留；如果 A 含 7，则这两个位置都会删除。它也不是按照 A 中出现次数去抵消 B 的多重集合减法，而是普通 membership filtering。
>
> 若业务要求 B 内部 unique，就要增加独立步骤，并明确保留第一次出现还是任意一次。排序、hash 或前序比较都会增加成本，也可能改变稳定顺序，因此需要更新 reference、输出语义和性能基线。

**可能追问：项目名称里说“去重”，面试时怎么避免歧义？**

> 我会先说清楚“删除 B 中已存在于 A 的元素”，并用一个重复 B 值的例子确认语义。这样可以避免面试官误以为当前 Kernel 同时完成了 B 内 unique、排序或完整集合运算。

---

### 66. 如果必须输出完整 compact 列表，3.02 ms 还能成立吗？

> 不能直接沿用当前 3.02 ms。当前 Kernel 只向 Global 写每份 B 的 count 和 checksum，完整 survivor 保存在 Shared 中并被本 Kernel 消费。若输出完整列表，就会新增 Global payload store，甚至增加全局输出位置分配。
>
> 按当前 50%存活率，每份 B 多写 16×4=64 Bytes，全部 16777216 份 B 就多出 1 GiB payload。可以使用每份 B 预留 32 槽位、只写前 count 个的固定槽接口；也可以输出紧凑的全局变长数组，但后者还需要计数、prefix sum 或分配机制。
>
> 这些方案的字节数、同步和 launch 数不同，必须各自测量。不能只把原 benchmark 输出参数加一个指针，就认为性能口径保持不变。

**可能追问：固定槽位分配 32 个 int，是不是必须把 32 个都写满？**

> 不必。容量可以固定预留，但只写 count 个有效元素，由 count 告诉消费者有效范围。如果约定剩余槽位也需要清零，则又增加写入工作；应分清显存容量占用和实际写入流量。

---

### 67. 如果 duplicate ratio 是 0%、50%、100%，你预期什么变化？

> 重复率变化首先影响 survivor 数。0%重复意味着每份 B 全部 32 个值都保留，Shared compact 和消费工作较多；100%重复则没有 payload 写入和求和内容；50%是当前每份保留 16 个的构造。
>
> 搜索函数仍保留六次展开迭代，但 active 状态和命中路径会不同，所以不能说整个执行完全不变。有效元素集中、交错或随机分布，还会改变 predication、subwarp rank 和 Shared bank pattern。
>
> 另外，v7 的 B cp.async 无论重复率如何都要读取输入，而 compact store 是条件执行的。整体瓶颈可能变化，速度也未必严格随 survivor 数线性变化，因此应把重复率和有效位置模式作为两个实验维度。

**可能追问：100%重复是否可以提前直接返回 count=0？**

> 只有上游已提供可信的元数据，或者当前 Kernel 已经完成证明全部命中的工作时才能这样做。没有该信息就仍然要读取和查询 B；而提前退出还要保证其他线程所需的 collective 与 block 输出同步不被破坏。

---

### 68. 如果 A 已经有序，还需要 Bitonic Sort 吗？

> 如果上游明确保证每份 A 已经有序，就可以直接使用寄存器中的 A 做二分，省去当前 15 个 Bitonic stage。当前源码无条件调用 warp_bitonic_sort，因此需要单独增加 sorted-A 路径，而不是认为现有版本会自动跳过。
>
> 但要判断这个有序条件是否真的免费。如果为了让此 Kernel 更快而在上游新增排序，完整流程可能只是把成本挪到了别处；如果 A 原本就由有序结构产生，省去重复排序才是有效的系统优化。
>
> 在当前接近带宽上限的条件下，删除排序也不保证带来同比例时间收益。它可以降低执行压力，但最终仍需要搬运 A/B 和结果，具体变化应重新测量。

**可能追问：先检查 A 是否有序，再决定排序值得吗？**

> 需要付出额外比较和 Warp 汇总，并引入选择逻辑。对只有 32 个值且网络固定的任务，这个检查可能吃掉收益。若调用方本来知道有序性，通过接口传递条件通常比每次在 Kernel 内重新判断更直接。

---

### 69. 如果 key 从 32-bit int 改成 64-bit，会发生什么？

> key 从 int32 改为 int64 后，A/B 输入字节数翻倍，比较、Shuffle 和寄存器占用也要重新评估。当前 int4 是每线程四个 32-bit key；若还保持四个 64-bit key，就变成每线程 32 Bytes，不能继续当成一条 16-byte copy。
>
> 可以保持 4×8 映射，每线程分两次 16-byte 搬运；也可以改变每线程元素数和 B 分批方式。A 的每个值也需要 64-bit 跨 lane 传输，Shared compact 的容量和 bank 地址分布都会变化。
>
> checksum 还要单独定义溢出语义，并同步修改 host reference。算法上的 membership 仍可用，但资源、指令和流量都不是简单换个类型就能忽略，原来的时间和带宽模型需要更新。

**可能追问：流量翻倍，时间就一定翻倍吗？**

> 只有其他因素基本不变、仍由同一带宽限制时，才可作为粗略估计。64-bit 带来的指令、寄存器和 Shared 变化可能改变 occupancy 或执行瓶颈，因此必须重新测量。

---

### 70. 如果要迁移到 H100 或更新架构，`cp.async` 还是最佳选择吗？

> 我会先迁移相同的输入输出语义，再选择搬运方式。当前主线是在 sm_89 上围绕一个 Warp、一组 A 和四份 B 设计的，B 只有 512 Bytes，总体有大量独立小 group；新架构的更大搬运机制未必正好适合这种粒度。
>
> 候选方案可以包括继续使用直接向量读取、保留 cp.async，或者在重新组织更大 tile 后研究目标架构的异步搬运机制。但更复杂的描述符、同步和任务组织也有成本，不能因为机制更新就预判它更快。
>
> 当前包没有 H100 或其他新架构结果，所以我的结论是保留可比较 baseline，重新检查资源、SASS 和带宽，再决定推荐版本，而不是把 Ada 上的排名直接外推。

**可能追问：为什么不直接说 H100 用 TMA 就最好？**

> 因为是否合适取决于访问粒度、布局和计算重叠窗口。当前每个 group 数据较小，若为使用更大粒度搬运而改变 block 组织，收益和额外同步都要重新评估。工具和机制应服务数据流，而不是反过来决定任务。

---

### 71. 如果换到 RTX 3090，为什么某个版本可能反而更快或更慢？

> 绝对时间会受目标卡带宽和执行能力影响，版本相对排名还会受编译器调度、资源分配和指令混合影响。例如 v2 用更多独立搜索链换可调度工作，v7 用 Shared staging 换异步机会，这些交换在另一张卡上未必仍有相同收益。
>
> 我会先为 3090 建立一致的基线：相同 group 数、重复率、输入布局和输出语义，使用正确架构重新编译，并记录工具链。随后检查每个版本的 Event、DRAM、stall、资源和 SASS。
>
> 如果某个版本翻转排名，我会比较限制是否从带宽转向执行、Shared 或驻留，而不是仅归因于“3090 比较老”。附件没有这组数据，所以不会报告未经测量的结论。

**可能追问：如何判断是 GPU 差异还是编译器版本造成的？**

> 尽量控制工具链，并在可行时做交叉对照：同一源码、相同工具链分别编译两种架构，再对同一张卡比较不同编译器产物。保留编译命令和 SASS，避免把两个因素同时改变后只归因于其中一个。

---

## 工程与实验追问

### 72. 你为什么使用 CUDA Event，而不是 CPU `std::chrono`？

> Kernel launch 对 CPU 是异步操作，CPU 计时如果没有等待 GPU，主要测到提交开销；若每次都强制 device synchronize，又会把同步开销混进去。当前程序用同一流上的 CUDA Event 包住 50 次目标 launch，再等待 stop event，适合比较这一段设备执行。
>
> 它先 warmup 10 次并同步，再记录 start，全部计时结束后才做 D2H 校验。因此初始化和校验拷贝不在 Event 区间内。最终 `elapsed_ms/50` 是这一组连续执行的平均值，不是程序从开始到退出的耗时。
>
> 如果关心完整系统延迟，CPU wall time 仍然有用，但必须定义正确边界并等待最终结果。单 Kernel、提交开销和 E2E 是不同测量对象。

**可能追问：Event 平均值是否完全不含 launch 间隙？**

> 不能绝对保证。如果主机提交速度跟不上或中间存在其他调度因素，GPU 时间线上可能包含间隙。当前 Kernel 是毫秒级、连续 launch，这种影响相对较小；若研究很短 Kernel，应结合时间线或其他提交方式检查测量边界。

---

### 73. Nsight Systems、Nsight Compute 和 SASS 各自回答什么？

> Nsight Systems 用于看系统时间线：CPU 什么时候提交、GPU 什么时候执行、memcpy 与同步在哪里，以及多个 stream 是否真正重叠。它适合回答完整流程为什么有空洞或为什么 GPU 等 CPU。
>
> Nsight Compute 聚焦某个 Kernel，提供资源、带宽、发射、stall 和内存访问等指标；SASS 则展示编译后的指令，帮助确认宽加载、Shuffle、barrier 和异步等待的实际位置。
>
> 当前压缩包主要包含 Event 结果、NCU reports/details 和 SASS，没有可核验的完整 Nsight Systems 系统时间线。因此它能支撑 Kernel 分析，却不能单独证明上层 CPU-GPU 往返已经减少。

**可能追问：只用 SASS 能不能确定哪个版本更快？**

> 不能。SASS 可以解释指令和依赖，但实际时间还受数据返回、并发资源和硬件状态影响。当前 v8 提前 issue 的事实能从 SASS 看出，最终值不值得仍要通过正常 benchmark 判断。

---

### 74. 你如何证明优化不是测量噪声？

> 首先看差异量级和重复性。约 1.38× 的主要收益较大，但也要保持相同输入输出与正确性；v6/v7 的差异只有千分级甚至更小，必须保存逐轮数据，并检查测试顺序、温度和频率是否影响结果。
>
> 当前 benchmark.sh 输出五轮均值，没有方差或置信区间，也没有交错顺序。因此可以报告既有均值，不能把它直接当成极小收益已经通过显著性检验的证据。
>
> 我会采用重复交错或配对测试，比较平均差值和自然波动，并观察计数器变化是否符合预期。若差异不可区分，就写性能持平，同时保留实现机制的差异。

**可能追问：SASS 已经证明少了几条指令，为什么还需要统计？**

> 因为指令可能不在关键瓶颈上，也可能被其他工作完全隐藏。减少指令只能构成合理假设，不能证明在实际运行条件下省下了可测量时间；千分级结果尤其容易被设备状态淹没。

---

### 75. 为什么仓库 README 的稳定时间和 profiles 中某次 result 不一样？

> README 保存的是作者报告的五轮参考均值，profiles 中 result 是随后某次采集前的正常程序运行输出，details 则是 NCU 采集环境下的 Duration。这三类记录的用途和批次不同，不能要求它们所有小数完全相同。
>
> 例如 shared baseline 的 README 是 4.154 ms，result 是 4.463923 ms，details 的 Duration 是 5.03 ms。它们的差异不能只靠“NCU 有开销”全部解释，因为 result 本身也是独立 Event 测量，运行环境变化同样可能参与。
>
> 我会保留原始记录并标清来源；若要发布一套新的正式结果，就在统一条件下重新采集所有版本，同时更新表格和环境信息。当前整理文档不会把这些旧数字改写成新实测。

**可能追问：这些差异是否足以说明原来的性能结论无效？**

> 不应只凭绝对时间变化直接否定或肯定。算法机制和多版本趋势仍有证据，但精确加速比必须绑定一套可比测量。最有效的做法是补齐同批运行和波动数据，而不是挑选最有利的一组数。

---

### 76. 公开仓库能证明简历里的哪些话，不能证明哪些话？

> 压缩包能直接说明实现结构：固定 32、默认 1:4、两种暴力 baseline、Bitonic 排序、Shuffle 二分、compact、block 输出、int4，以及 v7/v8 的异步调度。它还提供正常 result、NCU details 和 SASS，支持核对对应版本的资源与硬件行为。
>
> 但证据完整度并不相同。README 给出五轮参考均值，包里没有完整逐轮样本；v9 有源码和 README 描述，却没有 profiles/v9 原始产物；94.9%被 README 关联到历史布局实验，主线 v7 附件是 94.75%。
>
> 上层集成、真实数据集 E2E、个人分工和线上收益不在这个独立项目中。面试时应根据实际经历补充这些证据，不能把 microbenchmark 自动升级成完整业务系统。

**可能追问：最稳妥的性能介绍应该怎么说？**

> “按仓库参考均值，shared brute-force 到 v7 约为 4.154→3.018 ms，约 1.38×；上传的主线 v7 NCU 报告 DRAM 为 94.75%，接近 95%。”这样每个数字都能对应其证据来源，也避免混用历史布局指标。

---

### 77. 面试官追问上层 E2E 加速，你应该怎样拆解？

> 我会先把系统总时间拆开：CPU 准备、H2D/D2H、同步等待、过滤 Kernel、其他 GPU 阶段和框架开销。约 1.38× 只直接对应过滤 microkernel，不能直接当作总流程加速比。
>
> 如果只有该 Kernel 变快，其他阶段不变，那么 E2E 受热点占比限制；如果还把候选留在设备端、消除 host filtering 或减少后续无效工作，优化范围就扩大了，E2E 可能超过单 Kernel 的倍数。
>
> 当前附件没有系统 before/after 时间线和数据集表，因此我能说明拆解方法，但不能根据它确认具体 E2E。正式回答应提供同数据集、同正确性要求下的阶段时间和完整流程时间。

**可能追问：如果过滤只占原系统 50%，单 Kernel 1.38×，E2E 大约多少？**

> 若其他时间完全不变，按 `1/(0.5+0.5/1.38)`，约为 1.16×。这是条件推算，不是项目实测；它用来说明为什么优化热点的倍数不能直接等同于系统倍数。

---

### 78. 如果让你现在继续优化，你会怎样安排优先级？

> 我会先完善实验边界：补不同重复率和命中位置、随机输入、N、完整输出与 block size，确认当前 1:4 推荐版本的适用范围。然后补充逐轮计时和明确的构建环境，减少极小差异的解释困难。
>
> 局部实现上，优先围绕已有 excessive Shared wavefront 做最小对照，例如调整 compact 地址映射；同时检查新增地址指令、Shared 占用和同步，只有整体时间也改善才算有效优化。
>
> 接下来进入真实消费链，评估融合、减少全局读写和 CPU-GPU 同步。当前接近 DRAM 上限，继续复杂化 cp.async 调度的潜在回报较低，除非 workload 或架构改变后出现新的证据。

**可能追问：如果只能做一个实验，你先做什么？**

> 我会先补随机有效位置和多种重复率的正确性及性能对照。当前输入过于规律，先确认算法和瓶颈是否在更广输入下成立，能帮助判断后面究竟该优化搜索、compact 还是输出。

---

### 79. 你怎么发现 CPU-GPU 迭代同步是上层瓶颈的？

> 这类结论应通过系统时间线得出，不能从本项目的 NCU 表直接推出。需要观察一轮迭代结束后，CPU 是否等待 GPU 结果，再做判断或过滤，然后才提交下一轮，GPU 是否因此出现空闲间隙。
>
> 我会结合 API trace、memcpy 字节和 host 分段计时，确认究竟是哪项结果触发同步；再比较设备端处理后，是否减少往返、host 等待和 GPU 空洞，并验证完整流程结果一致。
>
> 当前压缩包没有这组系统记录，因此这道题应回答分析方法和实际掌握的调用链，不能说附件已经证明某个同步占比或精确节省时间。若确有内部实验，再补对应数据集与报告。

**可能追问：看到时间线上有 cudaDeviceSynchronize，就能说它是性能问题吗？**

> 不能。同步调用的等待时间可能只是前面必要 GPU 工作的完成时间；要判断是否可优化，必须看它是否阻止了本可并行的工作，或造成不必要的数据往返。不能只删掉同步而破坏依赖和正确性。

---

### 80. On-Device Candidate Filtering 的核心设计是什么？

> 核心是让 GPU 产生的候选尽量继续由 GPU 消费。候选生成后，在设备端完成 membership filtering 和 compact，再把有效候选交给下一阶段，只将真正需要的最终结果或控制信息返回 host。
>
> 在这个项目中，Shared compact 后读取并求 checksum，是对“同 Kernel 内后续消费”的简化模拟；它没有实现完整的候选生成、缓冲池和多轮迭代调度。因此可以用它说明底层过滤机制，但系统设计需要另行接入。
>
> 真正实现时要解决 survivor 容量、变长位置、buffer 生命周期，以及跨 Kernel/stream 的依赖；还要与原 CPU reference 对齐。收益来自少搬运、少同步和少处理无效候选，而不是只把一个 CPU 循环翻译成 CUDA。

**可能追问：设备端过滤后仍需把所有结果拷回 CPU，还会有收益吗？**

> 可能仍有 Kernel 计算收益，但减少往返的系统优势会受限。应看 CPU 实际需要哪些信息、返回频率以及能否批量处理；只有在完整流程中量化拷贝和等待，才能判断最终价值。

---

### 81. 为什么简历中的 E2E 是 1.76x-2.26x，可能大于 Kernel 的 1.38x？

> 这两个数字描述的范围不同。Kernel 的约 1.38× 只比较固定输入输出的过滤执行；E2E 若同时消除了 CPU 过滤、传输或迭代同步，就减少了多个阶段，因此理论上可以获得更大的倍数。
>
> 但附件中的 1.76×～2.26× 只出现在问答叙述里，缺少对应数据集、完整调用链和 before/after 测量。不能根据这个压缩包把它认定为已验证的系统结果。若这是实际内部工作，需要准备原始时间、配置和正确性记录。
>
> 面试中可以解释为什么倍数不矛盾，但具体范围必须有证据。如果没有系统数据，我会只报告有项目记录支持的 Kernel 结果，不自行编造各数据集的收益来源。

**可能追问：如何证明 E2E 大于 1.38×不是统计口径变了？**

> 比较时要固定输入数据、算法质量、硬件和结束条件，并给出优化前后阶段时间。只有确认减少了哪些额外工作，且完整功能等价，才能解释较大的 E2E；不能一边少做业务工作，一边当成同一任务加速。

---

## 算法正确性深挖

### 82. 你怎样证明 Bitonic Sort 后每个 lane 持有正确的有序元素？

> 代码中的 warp_bitonic_sort 是展开的固定网络，共 15 次 xor_swap。每次先用 shfl_xor 取得配对 lane 的值，再根据 lane 位和网络阶段决定保留较大值还是较小值；这些成对交换逐层构成更大的有序段，最终得到按 lane 排列的升序 A。
>
> 讲正确性时要同时说明配对和方向。配对距离来自当前 xor mask，方向由多个 lane bit 的异或或最后一轮的 lane bit 决定，不能只看一个方向位就忽略伙伴应做互补选择。
>
> 对实现的验证还需要完整排序输出。当前 benchmark 通过过滤结果间接覆盖排序路径，但不是单独的排序测试；更强的检查应比较随机、升降序、重复和极值输入与 std::sort 的逐元素结果。

**可能追问：相等值会不会在 compare-exchange 中被丢掉？**

> 相等时无论保留哪一侧，数值都相同，比较交换仍应保留正确的值多重集合。但如果元素还带有原始索引或 payload，就需要一起交换，并另外定义是否要求稳定排序；当前 membership 只处理 int 值，不要求 A 的稳定排序。

---

### 83. A 中存在重复值会影响二分和过滤正确性吗？

> 不会破坏当前 membership 语义。A 排序后，即使某个值出现多次，只要二分命中其中任意一次，就说明这个 B 元素存在于 A，可以过滤掉。当前实现不需要知道出现次数，也不需要先把 A unique。
>
> 但是需要分清算法适用性和现有测试覆盖：init_input_kernel 构造的 A 是 32 个互异偶数，没有覆盖重复 A。要验证完整范围，应加入重复区间、全相等和首尾命中等测试，并比较 CPU reference。
>
> 如果业务变成按出现次数抵消，例如 A 中一个 7 只能删除 B 中一个 7，那么现有 bool membership 就不够，需要统计和分配匹配次数，这是不同问题。

**可能追问：A 全部相等时，二分还有必要吗？**

> 如果上游已经知道该性质，可以专用化成一次值比较；若需要 Kernel 自己检查全相等，检查本身也有成本。当前通用路径仍应正确处理，但是否提供专用 fast path 要看这种输入是否常见。

---

### 84. 通用二分为什么使用 `[left, right)`，循环不变量是什么？

> 当前通用搜索初始区间是 left=0、right=32，即半开区间 [0,32)。在尚未命中时，如果 target 确实存在，至少有一个可能位置仍留在当前区间中，这就是维护的核心不变量。
>
> 每轮读取 mid：相等就设置 existed；target 较大则 left=mid+1，较小则 right=mid。没有命中时区间继续缩小，空区间表示不存在。代码仍固定执行六轮 Shuffle，但用 active 控制状态更新，非 active 时把 mid 设为 0，避免使用无效位置。
>
> 要注意，命中后程序并不继续收缩到 left==right，所以“最终所有线程区间都为空”不是这份函数的不变量。对目标 0，中点会走 16、8、4、2、1、0，正好说明为何本实现覆盖最坏路径需要六轮。

**可能追问：target 大于所有 A 时，left=32 后会不会 Shuffle 越界？**

> 不会。此时 left==right，active 为 false，mid 被设成安全的 0；该 lane 仍参与 full-mask Shuffle，但返回值不再用于搜索状态更新。这个安排同时照顾了边界安全和 collective 参与条件。

---

### 85. 二分搜索中的 Warp divergence 严重吗？

> 不同 B 值会产生不同 mid，但数据不同不等于一定发生控制流分叉。Shuffle 允许每个 lane 指定不同源 lane，所以 pivot 地址差异本身不要求 Warp 分开执行。
>
> 当前函数把六轮循环展开，所有 lane 都执行 Shuffle，再根据 active 和比较结果更新本地状态，编译器可以把一部分判断转成 predication。不过不能因此说整个 Kernel 完全没有 divergence：v7 details 的 Source Counters 仍报告了分支相关统计，compact 和其他条件也会参与。
>
> 我的判断是，这种固定结构降低了动态退出对 Warp 通信的复杂度，但实际分支成本需要看 SASS 和采样位置。不能仅凭 CUDA 里有 if 就断言严重，也不能仅凭固定轮数就宣称为零。

**可能追问：命中后 break，不就能省下后面几轮了吗？**

> 可能减少部分线程的逻辑工作，但会使各 lane 的执行路径和参与 mask 更复杂。由于其他线程仍可能需要从已命中线程持有的 sorted A 取值，随意退出会破坏 full-mask Shuffle 的条件；当前选择保留参与，只停止状态更新。

---

### 86. Compact 为什么能够保持 B 的原始顺序？

> v7 每个 B 元素的原始位置是 `sub_lane * 4 + item`。每线程的四个 valid bit 被放到 valid_mask 对应位置后，rank 就是该元素之前所有有效 bit 的数量。
>
> 例如 B 的位置 1、4、7 有效，它们前面的有效元素数分别为 0、1、2，于是写入 compact[0]、compact[1]、compact[2]。对任意两个有效位置 i<j，都有 rank(i)<rank(j)，因此输出保持原 B 顺序。
>
> 这个性质来自 rank 的定义，不依赖哪个线程先执行 store。每个有效元素目的位置唯一，故不需要 atomic 分配；但 Shared 写完后，消费者仍需要同步才能安全读到数据。

**可能追问：count 和 checksum 能验证这种稳定顺序吗？**

> 不能充分验证，交换两个有效元素不会改变 count 或 sum。稳定性应由 rank 逻辑证明，再用完整输出做逐元素检查；当前 benchmark 的校验结果不能替代这种顺序验证。

---

### 87. `valid_mask` 的构造为什么不会把四个 B 混在一起？

> 当前每个 Warp 的 b_id=lane>>3，将 32 lanes 分为四组；sub_lane=lane&7，表示 B 内的线程位置。每线程先生成四位 local_valid，再左移 sub_lane×4，让八个线程的 bit 区间互不重叠。
>
> 随后用对应的 subwarp_mask 和 width=8 做 OR reduction，最后把分区 leader 的 mask 广播给本分区。mask 指定参与线程，width 限定分区内源 lane 的解释，两者共同形成每份 B 独立的 32-bit 有效位图。
>
> Shared 地址又使用 `(warp_id * 4 + b_id) * 33` 分配独立列表区域。因此逻辑位图和写入区域都不会把不同 B 混在一起；但不同地址仍可能撞 bank，这不影响逻辑隔离。

**可能追问：为什么 A 的搜索使用 FULL_MASK，B 的 mask 合并却用 subwarp_mask？**

> 因为 A[32] 分布在整个 Warp，任意 target 都可能查询任意 A lane；B 的有效位和 checksum 只需要在负责同一份 B 的八个 lane 内汇总。这两种通信范围不能互相替换。

---

### 88. 为什么不用 CUB 的 BlockScan/WarpScan 做 compact？

> CUB 可以作为可靠的通用对照，但当前数据恰好只有 32 个有效位，每线程四元素，能够用一个 uint32 mask 表示整份 B。popcount 同时给出总数和元素前缀 rank，状态紧凑，也容易保持原顺序。
>
> 若改用 WarpScan，需要明确逻辑 warp 大小、每线程元素顺序和临时存储，并比较编译后的指令、Shared 和同步成本。对于这个固定形状，手写 mask 方案是合理选择，但不能没有实测就说一定比 CUB 更快。
>
> 压缩包没有 CUB 对照实现，因此当前只能解释设计依据，不能给出与 CUB 的加速比。长度或每线程元素数变化后，通用 primitive 的维护优势可能更明显。

**可能追问：已有 v1.5 的 register gather 能算 CUB 对照吗？**

> 不能。它是使用位选择和 Shuffle 的另一种手写 compact 路径，与 CUB 的实现和接口不同。它能验证“先 gather 再连续写”的假设，但不能替代真正的 CUB benchmark。

---

### 89. 为什么 checksum 可以防止编译器把 compact 路径优化掉？

> checksum 让 survivor 的值影响最终写入 Global 的可观察结果，因此编译器不能把所有输入相关计算都删成无关常量。当前源码还显式从 Shared compact 读取再归约，用它模拟后续消费。
>
> 但严格地说，最终只观察 sum，并不能从语言语义上强制保留每一次 Shared store/load。只要编译器能证明等价，就可能用另一种方法计算相同输出。因此“有 checksum 就绝对不会优化掉 compact”说得过满。
>
> 对本项目，应再检查实际 SASS：附件 v6/v7 确实保留了 Shared 相关指令，NCU 也记录了 Shared wavefront，说明当前编译产物执行了这条路径。正确性方面仍然需要承认 sum 的碰撞和顺序不可见问题。

**可能追问：加 volatile 就能解决 benchmark 的所有可信度问题吗？**

> 不能。volatile 可以约束特定访问被优化的方式，却也会改变指令和性能，且不替代同步或完整校验。应根据实验要保留的工作定义可观察输出，再用 SASS 检查，而不是把 volatile 当成通用保证。

---

### 90. 如果元素值溢出 `int`，checksum 会不会错误？

> 当前确定性输入的数值较小，16 个 survivor 的和在 int32 范围内，所以现有 checksum 构造是安全的。但如果 key 覆盖完整 signed int 范围，多个合法值相加就可能溢出，校验结果会变得不可靠。
>
> 若仍使用 int32 key，checksum 可改为 int64，并让 host reference 采用同样的提升和求和方式；或者明确使用无符号模加作为校验语义。不能依赖 signed overflow 的偶然机器结果来定义正确性。
>
> 这项修改不改变 membership，却会改变输出宽度、部分寄存器与规约指令。若保留为正式 benchmark，应更新 logical traffic 和结果接口，不能继续不加说明地沿用原来的 672 Bytes/group。

**可能追问：更宽的 sum 是否就能完全证明输出正确？**

> 仍然不能。它减少溢出问题，但排列错误不改变和，不同错误集合也可能产生同样的 sum。需要完整逐元素比较，或者多种校验摘要配合，但摘要仍不是无碰撞证明。

---

## CUDA 执行模型追问

### 91. 一个 block 为什么设为 512 threads，而不是 128 或 256？

> 512 threads 对应 16 个 Warp，默认一个 block 处理 16 个 group，最终产生 64 组 count/checksum。这样输出阶段由前 64 个线程连续写回，正好是两个 Warp 的连续结果。
>
> 在目标设备上，三个这样的 block 对应 48 个驻留 Warp，符合理论 occupancy 上限；当前 v7 的资源也允许这个驻留数。但这只能说明配置合理，不能证明 512 一定优于 128 或 256。
>
> 更小 block 可能改善调度粒度和 barrier 等待，也会改变输出汇总批次。当前 v5/v6 主体已将 BLOCK_SIZE 参数化，适合做对照；较早版本使用固定 constexpr，不能给所有版本直接传同一个宏就假设完成公平测试。

**可能追问：只改 block size，要不要改 grid？**

> 要保持总 group 数不变。v5/v6 使用 TOTAL_GROUPS/NUM_WARPS 计算 grid，因此默认 512 对应 262144 blocks。推广到不能整除的规模时还要补尾部处理，不能截断或越界，同时保证 block 输出同步合法。

---

### 92. `__launch_bounds__(BLOCK_SIZE)` 有什么作用？

> `__launch_bounds__(BLOCK_SIZE)` 告诉编译器这个 Kernel 允许的最大 block 线程数，帮助它在寄存器分配和目标执行条件下做选择。当前只有第一个参数，没有声明最少驻留 blocks/SM。
>
> 它不负责实际 launch，也不保证占用率。真正的线程数仍由 <<<grid, block>>> 指定，实际驻留还受寄存器、Shared 和设备上限共同约束。launch 超过声明的最大线程数会违反条件。
>
> 修改 BLOCK_SIZE 时需要同时考虑 Shared 数组、NUM_WARPS、grid 和输出覆盖，并重新查看 ptxas 资源。不能把 launch_bounds 理解为一个会自动寻找最优线程配置的开关。

**可能追问：加第二个 minBlocksPerMultiprocessor 参数一定更快吗？**

> 不一定。它可能让编译器为目标驻留数压缩寄存器，却引入更多指令或 spill。当前配置已经允许三个 block，是否增加约束应先确认实际资源瓶颈，再比较编译产物和运行时间。

---

### 93. `__restrict__` 在这些指针上为什么有用？有什么前提？

> 这里的 restrict 是提供给编译器的别名约束。当前输入 A、B 和两个输出数组分别独立 cudaMalloc，优化器可以利用它们不发生相关读写别名的事实，减少保守的重复读取或允许合法重排。
>
> 它不是缓存指令，也不保证数据放在哪一级内存，更不能直接解释为“GPU 自动向量化”。实际作用取决于代码的数据依赖和编译结果。
>
> 调用方必须遵守对应的访问约束，尤其不能让输出覆盖仍被当作独立输入读取的区域。若未来改为 in-place 接口，就要重新检查函数内所有访问路径，不能继续无条件保留原来的 no-alias 假设。

**可能追问：两个只读输入偶然重叠，也一定违反 restrict 吗？**

> 不能把规则简单概括成任何地址重叠都非法，关键在具体对象的访问和修改方式。工程上当前接口采用独立 buffer 最清楚；若要支持重叠，应依据实际读写关系重新审查，而不是随意扩大原契约。

---

### 94. 为什么代码里需要 `size_t` 计算 group/global index？

> 全局地址计算需要避免中间乘法先按 32 位溢出。当前 B 有 4194304×4×32=536870912 个 int，元素下标尚在 signed int 正范围，但整个 B 字节数已经是 2147483648 Bytes，达到 2 GiB。
>
> 因此源码把 group_id 或相关总量定义为 size_t，并在乘法前完成类型提升，让全局元素和字节偏移按足够宽的类型计算。若先用 int 算完再转 size_t，溢出已经发生，后面的转换不能修复。
>
> block 内的 lane、sub_lane 和 rank 范围很小，保留 int 更直接；只有全局规模和地址层使用较宽类型。这样既保证扩展安全，也避免无必要地把所有小索引都变成 64 位。

**可能追问：当前规模不变，还需要这些宽类型吗？**

> 需要，因为字节计数已经触及 32-bit signed 边界，而且代码还涉及分配大小与不同维度相乘。明确类型可以防止维护时改一个计算顺序就出错，不能只看最终元素下标暂时没超限。

---

### 95. Warp-synchronous 编程为什么仍要写正确 mask 和同步？

> Warp 内通信也需要明确参与者和内存顺序。当前 A 排序、二分使用 full-mask Shuffle，所以没有真实查询工作的 lane 仍可能作为 A 数据源参与，不能在其他线程还需要它时随意退出。
>
> B mask 和 checksum 使用对应的 8-lane mask，Shared compact 完成后又执行 syncwarp。这里 mask 不是过滤返回值的装饰，而是 collective 的参与约定；width 则决定分区内源 lane 的解释。
>
> 现代线程调度下，不能仅凭“同一个 Warp”就省掉 Shared 生产者与消费者之间必要的同步。推广到 tail 或条件路径时，尤其要重新检查谁实际执行了同一 collective、源 lane 是否活跃，以及读写是否有顺序保证。

**可能追问：可以在分支内部直接用 activemask 代替 FULL_MASK 吗？**

> 不能机械替换。分支内观察到的活跃线程不一定等于算法需要的完整参与集合，也不能补回缺失的 A 源 lane。应先按数据通信设计参与范围，再安排控制流满足该范围，而不是事后用一个 mask 掩盖错误。

---

### 96. 四个 subwarp 使用不同 mask 时，会不会让一个 Warp 真正并行执行四条指令？

> 不会让一个物理 Warp 变成四个独立的 Warp 调度实体。当前四个 subwarp 仍执行同构的搜索和 compact 指令，优势是一条 Warp 指令的不同 lane 同时处理四份 B 的对应数据。
>
> mask 和 width 主要组织哪些线程交换数据，并不为每个 subwarp 创建独立的 instruction stream。若四组线程走不同控制分支，仍可能产生 divergence 和额外执行路径。
>
> 因此这里的“同时处理四份 B”应理解为数据映射上的并行分工。A 仍由整个 Warp 协作排序，B 每线程四元素仍逐 item 搜索，不能把四个 subwarp 说成四个独立 Warp。

**可能追问：为什么这种划分仍有意义？**

> 它让连续的四个 int 落在同一个线程，形成自然的 int4 或 16-byte copy，同时让八个线程覆盖一份 B。它改善搬运和本地状态组织，并不是增加了硬件线程总数。

---

### 97. `volatile Shared Memory` 在 v8 中解决了什么，为什么不应滥用？

> v8 的 volatile Shared 数组是为了让 A 的 staging 访问在编译时保持实际存在，并与 block barrier、后续读回形成调度约束。它在排序前保存并读回 A，排序后又保存并读回 sorted A，帮助把 B issue 和 wait 分别放在目标位置。
>
> 这属于明确的调度实验，不是数据正确性本身必须采用的算法。代价包括额外 STS/LDS、2048 Bytes 数组和一次额外 block barrier；上传的 SASS 可以看到这些操作保留下来。
>
> volatile 不是跨线程同步，也不代表原子访问或完整 fence。当前代码仍要通过适当同步保证依赖；滥用 volatile 会限制编译优化，不能作为所有性能问题的通用解决方案。

**可能追问：为什么排序后还要再做一次 Shared round-trip？**

> 只把 B issue 提前，并不能自动保证 wait 不被调到排序完成前。排序后的写回和读回给后续使用增加实际数据路径，用来约束等待附近的调度；它也增加成本，所以需要与扩大窗口的收益一起评估。

---

### 98. inline PTX 使用 `memory` clobber 能保证硬件执行顺序吗？

> 不能保证任意硬件执行都严格按源码顺序。asm volatile 主要影响编译器对 asm 的处理，memory clobber 表达相关内存影响；它们不等于 GPU memory fence，也不能替代真正的线程同步或异步 copy 完成等待。
>
> 本项目的 cp_async_16、commit 和 wait helper 已经使用了 volatile asm 和 memory clobber，但 v7 的 LDGSTS 仍被调到大部分排序之后。这说明没有必要依赖的寄存器计算与搬运仍可能被重新安排。
>
> 若要表达正确性约束，应使用数据依赖和规定的同步机制；若是研究性能顺序，则可以像 v8/v9 那样做独立实验，并检查最终 SASS。不能仅因 inline PTX 看起来更底层，就忽略编译器后续阶段。

**可能追问：那在 CUDA 源码里把 cp.async 写到最前面有什么用？**

> 它表达了希望发射的操作和依赖关系，但不是强制所有无关操作排序的指令清单。是否保留足够独立计算窗口，需要检查机器码；如果已经能满足吞吐，强行约束调度还可能增加成本。

---

### 99. 为什么 `cp.async.wait_group 0` 后不一定需要整个 block 的 `__syncthreads()`？

> 当前 B staging 的所有权很简单：线程 tx 把自己的 16 Bytes 搬到 s_b4[tx]，之后还是线程 tx 读取这个位置。wait_group 0 等待该线程此前提交的异步组完成，因此不需要为了这次同线程消费再增加整个 block 的 barrier。
>
> 这与后面的结果输出不同，后者由不同 Warp 的线程读取各 leader 写入的 Shared，所以仍然需要 syncthreads。不能因为 B staging 不需要 block barrier，就推导出整个 Kernel 不需要 block 同步。
>
> 如果以后改成少量线程负责搬运、整个 Warp 或 block 协作消费，等待必须覆盖所有生产者，并建立相应可见性。每线程 wait 的作用域要按 PTX 语义理解。[PTX cp.async.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group)

**可能追问：一个线程执行 wait_group 0，能代表其他线程的 copy 都完成吗？**

> 不能。它等待的是当前线程所提交的异步组。协作 tile 需要设计所有生产者的完成与消费者之间的同步关系，不能只让 leader wait 一下就直接读取整个 tile。

---

### 100. Kernel 中两个 `__syncthreads()` 的代价如何判断？

> 首先要纠正数量前提：当前 v7 主路径有一次 block 级 syncthreads，用于结果输出；compact 后的是 syncwarp。v8 额外增加一次 block barrier，因此 v8 才是两次。分析成本应针对具体版本。
>
> barrier 的代价包括同步指令和到达不均衡造成的等待。输出 barrier 是因为前 64 个线程要消费整个 block 的结果数组；v8 的额外 barrier 则属于强制调度方案的一部分，二者目的不同。
>
> 我会结合 barrier stall、block 大小和正常时间判断，而不是按“一个 barrier 固定多少纳秒”直接相加。v7/v8 同时还改变了 Shared 访问，所以它们的时间差也不能全部归给 barrier。

**可能追问：如果把 v7 最后的 barrier 删除，会发生什么？**

> 前 64 个线程可能在其他 Warp 尚未写好结果时就读取 Shared，产生竞态。要消除它，必须同时重设输出所有权，例如让各 Warp 自己完成写回，并重新比较访问合并性；不能直接删除必要同步。

---

## 内存系统与吞吐追问

### 101. 你说 DRAM 接近 95%，A/B 数据会不会其实大量命中 L2？

> 仅从工作集看，A 为 512 MiB、B 为 2 GiB，明显大于该卡可容纳的 L2 工作集，顺序处理大量不同地址并不能让整个输入长期留在 cache。A 的查询复用主要发生在同一 Warp 的 sorted A 寄存器中。
>
> 不过工作集大不能证明每个 load 都 miss，因此还要看实际报告。上传的 v7 details 中 L2 Hit Rate 约 4.79%、L1/TEX Hit Rate 为 0，DRAM 为 94.75%，与大规模流式访问相符。
>
> 这些是该次采集的统计，受到测量配置和访问混合影响，不能无限推广到更小输入或真实系统中的缓存复用。若输入缩到 cache 可容纳范围内，主要瓶颈和版本差距都可能改变。

**可能追问：重复 launch 50 次，不就能把输入热到 L2 里了吗？**

> 重复执行能形成稳定状态，但每次仍会遍历远大于 cache 容量的工作集，不能同时缓存全部输入。应结合重用距离和实际 hit rate 判断，而不是只要重复运行就假设所有数据都命中。

---

### 102. RTX 4090 理论显存带宽怎样估算？为什么和 NCU 的 peak 不一定相同？

> 规格带宽可以用数据率乘总线宽度换算：21 Gbit/s×384/8，约为 1008 GB/s。这里使用的是规格级数据率，不应再随意多乘一次 DDR 系数，也要区分 bit 和 Byte。
>
> NCU 百分比采用对应的峰值及时钟口径，与这个规格分母未必相同。附件 v7 的带宽是 930.73 GB/s，DRAM Throughput 为 94.75%；若除以 1008，约为 92.3%，不能把两个比例当同一个计算。
>
> 实际判断还应结合设备当时频率、读写混合和访问效率。本文使用附件报告中的 94.75%描述该次主线采集，不重新用规格数字推造一个 NCU 百分比。

**可能追问：接近 95%意味着最多只能再快 5%吗？**

> 只能在相同字节量、相同峰值口径等条件下作非常粗略估计，不能直接作为严格收益上限。如果能减少输入输出字节、改变复用或融合阶段，时间仍可能明显下降；百分比只描述当前访问和测量条件。

---

### 103. Coalesced access 的判断单位是什么？是不是相邻线程相邻地址就结束了？

> 判断 coalescing，应从同一 Warp 内同一条内存指令的活跃 lane 地址出发，同时考虑每线程访问宽度、对齐和边界。多个 Warp 最终写入相邻地址，不等于它们的请求会自动合成一条 Warp 指令的请求。
>
> 当前 v6 的 B load 每 lane 16 Bytes，32 lanes 覆盖连续 512 Bytes，地址利用率较好；v1 的输出则每轮只有 lane0 写一个结果，即使不同 Warp 的结果在数组里相邻，也仍然是各自发出的稀疏 store。
>
> v5 将 block 结果汇总到 Shared，再由连续线程写回，正是改变了同一 Warp store 的参与线程和地址组织。coalescing 应针对读写指令分别分析，不能一句“数据连续”概括整个 Kernel。

**可能追问：Shared bank conflict 也属于同一种 coalescing 问题吗？**

> 它们都与地址组织有关，但机制不同。Global 主要看请求覆盖的 sector/cache line，Shared 主要看 bank 和 wavefront；改好了全局连续加载，并不保证动态 compact 写入也没有 bank conflict。

---

### 104. 128-byte B 为什么还强调 16-byte 对齐，而不是只要 128-byte 对齐？

> 当前每线程指令搬运 16 Bytes，因此它的基本合法条件是每线程起始地址满足 16-byte 对齐。B 列表大小是 128 Bytes，这描述整份列表的字节数，不是说每个线程都从 128-byte 边界开始。
>
> 例如第一个 B 的各 lane 地址是 base+0、16、32……112。基址 128-byte aligned 时，这些地址都满足 16 字节对齐，但后七个地址并不各自满足 128 字节对齐，也完全不需要。
>
> 更强的列表基址对齐有利于请求边界和覆盖效率；指令所需的最低对齐则关系到访问是否合法。两者作用不同，不能混成“只要列表大小是 128，就能从任何地址做 int4”。

**可能追问：基址只有 16-byte 对齐，功能是否还能正确？**

> 只要全部源和目标访问满足 16 字节对齐且范围合法，当前宽搬运可以满足对齐条件；但列表可能跨越更多请求边界，性能需要重新评估。正确性所需对齐和最理想的事务边界不是同一个要求。

---

### 105. v5 的 coalesced store 为什么提升很小？

> 因为当前 Global 输出只是 count 和 checksum，每份 B 共 8 Bytes，总计 128 MiB；相比 A+B 的 2.5 GiB 输入，它约占逻辑流量的 4.76%。优化输出事务有价值，但不是全部成本的主体。
>
> v5 具体沿用 v1 的搜索和 compact，每轮仍由 Warp lane0 产生结果，区别是先写 block Shared，最后由 tx<64 的线程连续输出。它还增加 512 Bytes 的结果暂存和一次 block barrier，因此改善 store 效率的同时也付出额外成本。
>
> README 的均值从 v1 约 3.031 ms 到 v5 约 3.025 ms，变化较小符合当前瓶颈状态。但不能把 4.76%当严格加速上限，因为原稀疏事务的硬件成本不只等于有效字节数，仍要看实际请求。

**可能追问：如果改为完整 survivor 输出，这个优化会更重要吗？**

> 可能，因为 payload store 的占比会上升。但输出格式、地址分配和保序要求也会改变。需要为完整输出设计新的写回方案，并测量真实请求与时间，不能直接把当前 count/checksum 的收益外推。

---

### 106. 为什么减少指令后 DRAM 流量没变，时间却能下降到带宽上限？

> 在尚未饱和的情况下，内存系统需要 Kernel 持续提供足够请求。若 Warp 长时间被搜索相关指令、队列压力或依赖限制，后续 group 的推进速度会受影响，DRAM 即使有空余能力，也未必获得足够可处理的请求。
>
> 算法和数据路径重构能改变这种供给，让同样字节量在更短时间内完成。不过本项目要纠正一个前提：v0 到 v7 的总 Executed Instructions 并没有下降，附件分别约为 16.23 亿和 19.37 亿。减少的是某些搜索轮次和相关压力，不等于全部机器指令数更少。
>
> 因此更准确的解释是“指令组合、依赖和请求推进方式改善”，让带宽更充分利用；达到 DRAM 上限后，继续减少某些局部指令就未必缩短时间。

**可能追问：为什么更多总指令反而可以更快？**

> 不同指令的吞吐、延迟和资源竞争不同，而且有些能与其他工作重叠。总条数不反映关键等待。当前 v0 的 MIO 队列等待很高，优化版即使有更多边界、mask 和地址指令，也可能更快推进整个任务。

---

### 107. 这个 Kernel 的 arithmetic intensity 应该怎么算？

> 通常 arithmetic intensity 是操作数除以数据流量，但这个 Kernel 主要做整数比较、Shuffle、位运算和地址计算，不适合直接套用 FP32 FLOP/Byte。附件的浮点 roofline 也显示 FP32/FP64 利用率为 0，不能据此认为 Kernel 没有执行压力。
>
> 可以自定义 comparisons/byte 或主要 Warp stage/byte，但必须说明操作口径。暴力每组最多执行 4×32×32=4096 个元素比较；排序加二分的 15+4×6=39 是主要 Warp 通信 stage，两者不能直接当成同一种操作相除。
>
> 字节分母也要区分程序的 672 logical Bytes/group 和实际 DRAM bytes。最终我更倾向用 NCU 的分管线、队列、内存和正常时间做判断，同时用算法步骤解释为什么搜索结构改变。

**可能追问：能不能用 4090 的 FP32 峰值估算这个 Kernel 的算力上限？**

> 不能直接这么做。比较、Shuffle、popcount 和 Shared 访问不按 FP32 峰值执行，瓶颈还包括队列与依赖。要建立上限模型，应选相应的操作吞吐和实际内存带宽，而不是把所有指令都当浮点运算。

---

### 108. 如果把 A 缓存在 Shared Memory 供多个 Warp 复用，会更快吗？

> 当前一个 group 对应一份独立地址的 A，一个 Warp 负责该 group，并已经在寄存器里复用 A 处理四份 B。不同 Warp 在输入契约上没有共享同一个 A 对象，所以把各自 A 放进 block Shared，并不会自动产生跨 Warp 复用。
>
> 虽然测试生成器让所有 group 的 A 数值相同，但这是确定性输入的特征，不是 Kernel 接口承诺。不能据此只读取一份 A 给整个 block 用，否则换成不同 A 的合法输入就会错误，也相当于利用 benchmark 特例少做工作。
>
> 如果真实业务确实多个 group 共享同一个 A，可以重新设计 block 级缓存，比较减少 Global 读取与新增 Shared、同步的成本。那应作为新的共享输入契约，而不是直接套用当前性能。

**可能追问：v0_shared 是否已经证明 Shared 缓存永远没用？**

> 没有。它只是把每个 Warp 自己的 A 用 Shared broadcast 提供给暴力搜索，并未创造跨 group 复用。该实验说明更换存储位置没有消除重复搜索成本，不能否定真正共享 A 的其他场景。

---

## Profiling 陷阱与实验答辩

### 109. NCU 的 Speed of Light 指标能直接相加吗？

> 不能直接相加。Compute、Memory、DRAM 和 L1/TEX Throughput 都是各自相对峰值的利用指标，其中还存在汇总与子指标关系，不是互斥的时间占比。
>
> 本项目 v0 的 Compute 和 Memory 都约 97.85%，DRAM 却只有 56.19%，同时 L1/TEX 约 97.90%。这并不矛盾，也不表示 GPU 做了超过 100%的工作；需要展开具体资源和 stall，才能理解高值来自哪里。
>
> 附件进一步指出 v0 的 MIO queue 等待和较高 Mem Pipes Busy，而浮点利用率为 0。因此不能把 Compute 97.85%直接翻译成“用了 97.85%的 FP32 算力”。SOL 用于定位重点，真正归因还要看对应 workload 和源码。

**可能追问：Compute 比 DRAM 高，就一定是计算瓶颈吗？**

> 不能仅凭两数大小判断。它们的峰值分母和汇总口径不同，还可能涉及访存管线。当前更稳妥的说法是 v0 主要受搜索相关执行与 MIO/L1TEX 路径限制，而不是 DRAM 已饱和。

---

### 110. 面试官说“94.9% 是工具估算，不代表真带宽”，你怎么回答？

> 我会先承认百分比是按工具指标定义归一化的结果，不能单靠一个接近 95%的数字结束分析。对上传的主线 v7，应准确引用 94.75%，并同时给出 Memory Workload 中约 930.73 GB/s 的硬件统计。
>
> 再结合大工作集、较低 L2 hit rate，以及 v1 之后多种优化时间接近的现象，形成接近 DRAM 上限的证据。程序约 934 GB/s 的 logical bandwidth 可以作为数量级对照，但不是另一次独立硬件测量。
>
> 如果对方进一步追问，我会补充 DRAM read/write bytes、时钟和相同访问条件的带宽校准。当前附件没有列出所有原始计数，因此不应自行编造精确的 DRAM 字节或额外校准结果。

**可能追问：为什么不直接沿用简历中的 94.9%？**

> 因为 README 将它关联到历史 lane-major 实验，而当前主线 v7 details 明确是 94.75%。概括可以说接近 95%，精确展示则应带版本和来源，避免把不同实验的数字混在一起。

---

### 111. NCU replay 会不会改变 Kernel 行为？

> NCU 可能需要多次执行目标 Kernel 来收集不能同时获取的指标，测量还涉及缓存、时钟和保存恢复等机制，因此采集环境不一定等同于正常运行。这里不能把 profiler Duration 直接替代 Event benchmark。
>
> 当前过滤 Kernel 只读取固定 A/B，写独立结果，不依赖每次 launch 累积不可逆状态，适合做这类采集。脚本先用 kernel name 选择目标，再跳过匹配的 warmup，只采一次目标 launch，但该 launch 的 full 指标仍可能由多个 pass 组成。
>
> 我会把正常性能与 profile 行为分开报告，同时检查输入输出在采集方式下是否可重复。对于含跨进程通信或依赖 host 响应的 Kernel，还需要另外选择合适的 profiling 方式。

**可能追问：一次 NCU 报告里有很多指标，是否代表它们全在同一时刻测得？**

> 不一定，可能来自不同 pass 的组合。因此需要稳定、可重复的 workload；若任务行为变化很大，简单比较汇总值会有风险。当前独立确定性 benchmark 有助于稳定采集，但仍不能抹平采集环境和正常运行的区别。

---

### 112. 为什么只 profile 一次 launch，不 profile 50 次平均？

> full 指标集合通常需要多 pass，如果对 50 次 launch 都采集，会显著增加时间和报告体积，而大量内容可能重复。当前 collect_profiles.sh 使用 kernel name 过滤、launch-skip 10、launch-count 1，目标是分析一个稳定状态下的目标调用。
>
> 正常 benchmark 则另外通过 Event 平均 50 次，外层默认五轮，用于性能统计。这两部分承担不同职责：一个解释单次设备行为，一个衡量正常运行时间。
>
> 如果怀疑 counter 不稳定，应有针对性地重复采集报告并比较，而不是认为把 launch-count 改成 50 就自动得到可信平均结论。还要控制每次采集的输入和环境。

**可能追问：为什么要按 kernel name 过滤后再跳过 warmup？**

> 因为程序还有初始化 Kernel。过滤可以让跳过和采集针对目标过滤函数，避免把初始化当成第一次 warmup 或采错对象。最终仍应检查 report header 的函数名、grid 和 block 是否符合预期。

---

### 113. 你怎样从 SASS 判断 `int4` 或 `cp.async` 真正生成了宽搬运？

> 先找到目标函数，不能在整个可执行文件里随便看到一条宽加载就当作 B 路径。附件 v6 的目标 Kernel 在偏移 0x04a0 有 `LDG.E.128.CONSTANT`，结合 B 地址计算可以确认直接向量读取；v7 在 0x0670 有 `LDGSTS.E.BYPASS.128`，之后 0x06e0 用 `LDS.128` 读取 staging。
>
> 再检查 A 的标量 load、B 的地址步长、issue/wait 和消费者，确认 .128 表示的宽操作确实属于预期数据。v7 的 Shared B 区域偏移为 0x2300，恰好等于前面 compact 与两个输出数组的 8960 Bytes，也能与源码布局相互印证。
>
> 这些地址是附件当前编译产物的定位点，不保证换工具链后不变。验证原则是数据流和操作数，而不是死记固定指令偏移。

**可能追问：源码写了 int4，为什么还可能要检查是否拆成标量？**

> 因为最终加载方式受使用模式、优化和编译条件影响。尤其只使用部分分量或存在别名、对齐问题时，源码类型不足以证明机器指令。当前包有实际 SASS，所以可以直接给出该次编译的证据。

---

### 114. 为什么一个 NCU estimated speedup 不能直接当优化收益？

> NCU 的 estimated speedup 是帮助发现可能优化方向的估计，不是已经实现的整体收益。它通常针对某项局部问题，并不自动计算修复需要新增的地址指令、同步、资源占用或新的瓶颈。
>
> 例如 v7 的 Shared store 提示有约 1.6-way conflict，相关规则给出较大的估计，但当前 DRAM 已经接近上限。即使通过 swizzle 降低 excessive wavefront，也可能没有明显缩短时间。
>
> 我会把建议转成一个可验证假设：保持其他条件，改一种地址布局，先确认目标指标改善，再检查正常时间和正确性。只有整体结果也有可靠收益，才称为优化成功。

**可能追问：estimated speedup 比理论剩余带宽空间大很多，怎么解释？**

> 说明它不是在当前所有约束下的严格 E2E 上限。不同规则聚焦不同资源，不能相加或直接与 DRAM 百分比作算术拼接；最终必须通过实测确认改善是否落在关键限制上。

---

### 115. 如何确认没有测到初始化 Kernel 或其他 CUDA 操作？

> 当前源码中初始化完成后先同步，warmup 也在计时区间外；Event 只包住目标 Kernel 的 50 次连续 launch，Host 的结果拷贝与校验在 stop 完成后才执行。
>
> 采集脚本还建立了版本到实际函数名的映射，使用 kernel-name-base function 和 regex 过滤，再跳过 10 次匹配调用。需要注意 v1.5 的真实函数名含 v2_register_gather_compact，而不是仅按文件名猜函数。
>
> 最后检查 details 的 header，例如 v7 应是 v7_subwarp_cp_async_kernel，grid 为 262144、block 为 512。源码边界、脚本过滤和报告对象三个方面对应，才能确认这个毫秒数测的是目标任务。

**可能追问：只看程序打印的版本标签够吗？**

> 不够，标签可能因历史重命名而保留旧称，也可能运行了旧二进制。应同时核对可执行文件、函数名、配置和构建依赖；附件 v1.5 的旧 V2 标签就是需要结合上下文识别的例子。

---

### 116. 版本很多，如何避免 benchmark 脚本把旧二进制当成新源码？

> 首先让构建系统追踪真正的源码依赖。v7/v8/v9 的文件只是宏配置入口，共同 include v6.cu，所以修改 v6 后，这三个目标也必须重编译。当前 Makefile 已经显式补了这三条依赖。
>
> 但它没有自动把命令行编译参数变化当成源码时间变化。修改 NUM_B_LISTS、BLOCK_SIZE 相关参数或 nvcc flags 后，旧目标可能仍被认为最新；仅执行 make all 不能保证参数已经进入二进制。
>
> 正式采集可以使用独立构建目录或受控重建，并记录源码版本、完整编译命令、工具链和二进制 hash。采集脚本会复制可执行文件到结果目录，这有助于保存产物，但仍应补齐产物与源码版本的对应信息。

**可能追问：换了 EXTRA_NVCC_FLAGS，为什么 make 可能不编译？**

> 普通 Make 主要根据依赖文件时间判断目标是否过期，不会自动记住上次的命令行变量值。可以把配置编码进目录或生成依赖文件；临时实验则明确重新构建，并确认输出的资源与配置。

---

## 工程决策与行为面追问

### 117. 为什么保留 v0 到 v9，而不是只提交最快的 v7？

> 版本链能展示结论怎样形成。v0/v0_shared 区分寄存器广播和 Shared 广播，v1 给出排序加二分的核心变化，v1.5/v2/v3/v4 验证 compact、ILP 和固定搜索的取舍。
>
> v5/v6/v7/v8 又逐步改变输出组织、线程映射和搬运调度。保留这些对照，别人才能检查哪些收益来自算法、哪些只是接近带宽上限后的局部试验，而不是只看一份最终代码和一个倍数。
>
> 工程上实验仓库与生产实现可以不同。生产只暴露经过验证的推荐接口，实验代码和 profile 则保留用于回归和跨架构分析。还应标注证据缺口，例如 v9 缺少随包提供的原始 profile。

**可能追问：版本这么多，会不会让面试介绍显得零散？**

> 口述时按三层组织即可：算法重构、数据路径、profiling 验证。重点讲 v1 的主收益，再用 v6/v7 和 v8 的一个案例解释后续取舍，具体版本留给追问时展开。

---

### 118. 如果团队要求删掉 v8/v9，你会怎样处理？

> 如果生产团队希望减少代码维护面，我可以让生产分支只保留推荐实现和必要 fallback，把 v8/v9 的实验目的、关键差异、性能记录及机器码证据移到实验目录或单独文档。
>
> 这不会丢失核心知识：v8 说明可以通过 Shared 和 barrier 改变调度窗口，v9 探索 noinline 方式，而扩大窗口不一定改善最终吞吐。需要留下的是可追踪结论，而不是必须让所有实验一直参与默认生产构建。
>
> 同时会保留正确性和性能回归入口，避免删除实验时误删公共 v6 主体中的必要分支或影响 v7。实际调整应依据团队维护要求，不以“负优化没有价值”作为唯一理由。

**可能追问：既然 v9 没有完整原始报告，应该怎么归档？**

> 保留源码和 README 已有记录，并明确标注原始 profile 未随当前包提供。若以后补采，就把工具链、二进制和报告一起补齐；不能用 v8 的 SASS 替代 v9 的证据。

---

### 119. 如果同事质疑 0.05% 的 `cp.async` 优化不值得合入，你怎么回答？

> 我会承认这个质疑合理。当前参考均值的差别非常小，benchmark 又没有给出方差或置信区间，单凭约 0.05%的说法不足以证明值得增加 inline PTX 和 Shared staging 的维护成本。
>
> v6 已经提供直接 int4 的简单路径，v7 则验证异步搬运和编译调度，研究价值明确；生产是否采用 v7，应看统计收益、架构兼容和整体代码复杂度，而不是把“用了 cp.async”当目标。
>
> 无论最终选 v6 还是 v7，项目的主要性能成果仍来自排序加二分。面对质疑，我会把主收益和实验性变化拆开说明，而不会放大几乎持平的时间差。

**可能追问：如果面试官问为什么 README 仍推荐 v7，怎么回答？**

> README 推荐的是该实验链中保留的实现，参考均值略低且有对应 profile；但推荐不等于已证明对所有环境都显著优于 v6。实际工程决策可以在同等性能下优先选择更易维护的直接读取方案。

---

### 120. 你遇到的最大失败假设是什么，学到了什么？

> 一个代表性失败假设是：只要把 B 的异步搬运更早放到 A 排序前，扩大 overlap，就能继续明显加速。v7 的 SASS 先表明源码顺序并没有形成设想中的长窗口，v8 又通过显式 Shared 和 barrier 把 issue 提前。
>
> 结果是 v8 的指令顺序达到设计目的，整体时间却没有明显变好。说明当前可能已经有足够的延迟隐藏，加上 DRAM 接近上限，额外同步和搬运成本抵消了潜在收益。
>
> 我得到的教训是分开验证三件事：优化假设是否合理，编译器是否生成预期行为，预期行为是否改善实际瓶颈。前两项成立，不代表最后一项必然成立。

**可能追问：这能否证明任何异步搬运都没有价值？**

> 不能。它只说明当前小 group、数据量和驻留条件下，强制扩大该窗口没有明显回报。换成更大独立计算、更长流水或其他访问模式，收益可能不同，必须在对应场景重新实验。

---

### 121. 如果面试官让你现场设计一个新优化实验，你怎么回答？

> 我会选择 compact Shared 地址映射做一个最小实验，因为 v7 已有 excessive wavefront 证据。保留输入、搜索、count/checksum 语义和 launch 配置，只对 compact 的物理布局设计一种可逆映射。
>
> 写入时把逻辑 rank 映射到物理槽位，读取时使用同一映射，先验证每份列表没有覆盖、顺序消费仍正确，再看目标 STS/LDS 的 excessive wavefront 是否下降。然后比较新增地址指令、Shared 用量和正常时间。
>
> 若 conflict 改善但时间持平，我会记录它改善了局部访问，却没有突破当前带宽限制；若更慢，则撤回生产路径。这个实验成功与否由整体性能决定，不由 NCU 提示是否变绿决定。

**可能追问：为什么一定要同时修改读取地址？**

> 因为 compact 的逻辑顺序没有变，只是物理存储重排了。若只改写入而仍按原连续地址读取，会读错值或未初始化槽位；checksum 偶然相同也不能证明这种不一致正确。

---

### 122. 如果新版本快 2%，但只在 100% duplicate 下有效，要不要采用？

> 先看生产中这种输入是否常见，以及有没有低成本识别方式。如果上游已有可靠的重复率或全命中元数据，可以为该场景保留专用路径；如果必须先完整搜索才能知道 100%重复，额外选择机制可能没有价值。
>
> 还要评估其他分布的回退表现、代码复杂度和统计可信度。只在极少见条件快 2%，却使常见路径变慢或难维护，不一定值得合入默认实现。
>
> 报告应明确写成“在 100%重复的测试条件下改善约 2%”，不能概括为全项目快 2%。当前附件没有这个实测，这是设计决策题，需要按假设条件回答。

**可能追问：能不能根据上一批数据的重复率选择下一批 Kernel？**

> 可以作为启发式，但数据分布可能变化，选择错误会损失性能。必须保证所有候选路径都正确，并把统计、切换和误判成本纳入 E2E；不能仅凭历史比例跳过必须的 membership 检查。

---

### 123. 如何向非 CUDA 面试官解释这个项目的核心价值？

> 我会先用一个简单场景解释：有一张包含 32 个编号的参考表，需要不断检查四张候选表，把已经出现过的编号剔除。原方案对每个候选都从头扫描参考表，重复做了很多查找工作。
>
> 我先把参考表整理成有序数据，再让四张候选表复用这个结果进行更短的查询；随后调整 GPU 线程如何读取和组织数据。按仓库参考结果，独立 Kernel 从约 4.15 ms 降到约 3.02 ms，约 1.38×。
>
> 最后通过硬件报告确认，现在已经接近显存供数能力，继续改小细节收益有限。对非 CUDA 面试官，重点是重复工作如何减少、结果如何验证，以及收益对应的范围。

**可能追问：如果对方问对业务到底有什么意义？**

> 我会说明它适合大量调用的小列表过滤热点，价值取决于在完整流程中的占比和是否减少后续工作。当前包证明的是底层 Kernel；要说明业务收益，需要实际调用规模和系统 before/after，不能只用单 Kernel 倍数代替。

---

### 124. 如何回答“这个项目是不是为了写简历而做的 microbenchmark”？

> 我会直接说明公开项目是独立 microbenchmark，它把固定 32、1:4 的过滤路径抽出来，保持输入输出条件，便于比较算法、线程映射和编译后指令。这是性能工程中用于隔离变量的一种合理方式。
>
> 它的价值在于源码、计时、正确性和部分 profile 能互相对应，尤其保留了不加速的实验；限制则是输入较规律，只输出 count/checksum，没有完整业务调用链和系统时间线。
>
> 因此我不会包装成完整生产系统。若确有实际集成经历，就另外讲真实接口、职责和系统数据；若没有，就把它作为专用 CUDA 优化项目，重点展示分析和验证能力。

**可能追问：microbenchmark 做得快，为什么不一定真实系统也快？**

> 真实系统可能有更小批量、不同重复率、缓存复用、完整输出和频繁 launch，也可能瓶颈在 CPU 同步或其他 Kernel。必须把该实现接入真实输入输出后测量，独立最优只是候选方案。

---

### 125. 如果面试官认为 1.38x 不够大，你怎么回答？

> 我会先解释优化对象和约束，而不是只争论倍数大小。这个项目保持 B-major 输入和固定输出语义，主要通过排序加二分改善执行路径，主线 v7 的 DRAM 已达到 94.75%，后续多种局部实验没有明显突破。
>
> 因此约 1.38×有具体的机制和硬件行为支撑，但它不意味着整个系统也提升 1.38×。实际价值还取决于调用频次、热点占比和能否减少后续数据处理。
>
> 如果希望更大收益，下一步应减少数据字节、融合设备端消费或处理系统同步，而不是继续把很小的 v6/v7 差异放大成主要成果。我会承认适用范围，同时解释当前约束下为什么结果合理。

**可能追问：是否应该为了更高倍数换一个更慢的 baseline？**

> 不应该。baseline 应代表明确且合理的原路径，并在比较前固定。这里有 register 与 shared 两种暴力实现，就分别报告对应倍数；不能在面试时临时挑较慢分母来制造更大成绩。

---

### 126. 如果让你重新做一次，这个项目流程会怎样改进？

> 如果重新做，我会先建立输入输出契约和完整正确性参考，覆盖随机 A/B、边界值、不同重复率、N 和稳定顺序，再开始性能实验。这样能避免最后才发现 checksum 没覆盖某类错误。
>
> 实验管理上，每个结果记录源码版本、编译参数、二进制、设备状态和逐轮时间，并明确区分 README 均值、result Event 与 NCU Duration。像 v9 这样的版本，应同时保存原始 profile 和 SASS，避免结论只有文字记录。
>
> 技术路线仍优先算法重构，再分析具体执行管线和数据路径；后续优化按潜在收益排序，及时停止没有统计收益的复杂化。最后再进入真实调用方，验证完整输出和 E2E，而不是让 microbenchmark 的结论越过自己的证据范围。

**可能追问：这次项目最应该保留的习惯是什么？**

> 保留“提出假设、做独立版本、验证正确性、测正常时间、用 NCU/SASS 解释”的流程。需要改进的是数据覆盖和结果追踪，让每个倍数、每条机器码结论都能找到对应产物。

---

## 项目核对说明

本版以本次上传的压缩包为项目快照，不将后续线上仓库变化混入结论。第 1～20 题及前面的项目介绍按要求保留原文；第 21～126 题已重新组织为面试回答与对应追问。下列说明补充保留部分中需要结合当前源码理解的口径。

- **1:N 复用与 subwarp 映射是不同层次。** v1 已经排序一次 A、顺序处理四份 B；v6/v7 的 4×8 映射进一步匹配连续四元素和 16-byte 搬运，并不是第一次实现 A 复用。
- **六轮搜索以函数实现为准。** 主线通用二分每轮判断相等，目标为最小值时可经过中点 16、8、4、2、1、0；v3/v4 才采用五次固定步长探测加最终比较。保留部分中的“最多五次比较”不应当作当前通用实现的精确描述。
- **1:1 的测试经历需要另外的实验记录。** 当前附件不足以独立核验保留部分中“测试过 1:1，两者接近”的实测声明。第 61 题给出的是设计分析，不将其改写成已经完成的实验。
- **v5 的原实现是整 Warp 顺序处理 B。** v5 由每个 Warp 的 lane0 暂存各 B 的结果，block 同步后由前 64 个线程输出；四个 subwarp leader 的结构属于后来的 v6/v7。
- **精确性能口径统一看第 31～34 题。** 主线 v7 附件 DRAM 为 94.75%；94.9%在 README 中关联历史布局实验。v0 result 的约 4.48 ms 是 Event 时间，NCU Duration 为 5.10 ms。v0 的报告明确指出 MIO 队列节流，不能只根据 Compute 高值就归因为 ALU 或浮点算力饱和。

## 附件证据索引

以下路径相对压缩包中的 warp-candidate-filter-main 项目根目录，便于对照源码与原始记录。

| 内容 | 对应文件 | 核对要点 |
| --- | --- | --- |
| 两种暴力基线 | src/v0.cu；src/v0_shared.cu | A 的供数方式、B-major、Shared compact、count/checksum |
| 主要算法重构 | src/v1.cu | 一次 A 排序，顺序处理多份 B，通用六轮搜索 |
| Register gather 实验 | src/v1.5.cu | fns 位选择、Shuffle gather、旧 V2 函数标签 |
| ILP 与固定搜索 | src/v2.cu；src/v3.cu；src/v4.cu | 两套搜索状态、五次探测加最终比较 |
| 合并结果写回 | src/v5.cu | block Shared 汇总，前 64 个线程输出 |
| v6～v9 公共主体 | src/v6.cu | 4×8、每线程四元素、mask/rank、Shared、异步宏分支 |
| 版本入口 | src/v7.cu；src/v8.cu；src/v9.cu | 宏定义后 include v6.cu，不是缺失实现 |
| 参考均值 | README.md | 作者报告的五轮均值，未附完整逐轮样本 |
| 正常 Event 记录 | profiles/各版本/*_result.txt | 采集前单独运行程序的平均时间，不能当成 NCU Duration |
| 主线硬件指标 | profiles/v0/v0_real_4b_ncu_details.txt；profiles/v7/v7_real_4b_ncu_details.txt | MIO、DRAM、Duration、occupancy、实际 Shared carveout |
| 宽加载与异步顺序 | profiles/v6/v6_real_4b.sass；profiles/v7/v7_real_4b.sass；profiles/v8/v8_real_4b.sass | LDG.128、LDGSTS、DEPBAR、LDS 与排序位置 |
| 构建与采集 | Makefile；scripts/benchmark.sh；scripts/collect_profiles.sh | 源码依赖、10 次 warmup、50 次 Event 平均、外层五轮、目标过滤 |
| 历史布局实验 | archive/transposed-layout/README.md 及同目录源码 | lane-major 与主线 B-major 分开，不含转换成本 |

v9 随包有源码与 README 记录，但没有 profiles/v9；完整系统 E2E、Nsight Systems 时间线和真实数据集记录也未随包提供。文中涉及这些内容时，分别作为既有文字记录、分析方法或下一步方案表述，不视为此次重新实测结果。



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
