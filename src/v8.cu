#define SUBWARP_USE_CP_ASYNC 1
#define SUBWARP_FORCE_EARLY_CP_ASYNC 1
#define SUBWARP_KERNEL v8_subwarp_early_cp_async_kernel
#define SUBWARP_VARIANT_LABEL "V8 4x8 Subwarp Early cp.async Candidate Filter"
#define SUBWARP_OVERLAP_LABEL "forced full A-sort window"

#include "v6.cu"
