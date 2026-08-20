#define SUBWARP_USE_CP_ASYNC 1
#define SUBWARP_FORCE_CP_BEFORE_A 1
#define SUBWARP_KERNEL v9_subwarp_cp_before_a_kernel
#define SUBWARP_VARIANT_LABEL "V9 4x8 Subwarp cp.async Before A Filter"
#define SUBWARP_OVERLAP_LABEL "noinline issue/wait boundaries"

#include "v6.cu"
