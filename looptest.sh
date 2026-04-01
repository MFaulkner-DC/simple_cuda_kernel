import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void test_loop(int *out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int val = 0;
    for (int i = 0; i < 100000000; i++) {
        val += i;
    }
    out[idx] = val;
}
""")

test_loop = mod.get_function("test_loop")

N = 160
out = np.zeros(N, dtype=np.int32)

test_loop(
    drv.Out(out),
    block=(N,1,1), grid=(1,1)
)

print("Kernel output:", out)
