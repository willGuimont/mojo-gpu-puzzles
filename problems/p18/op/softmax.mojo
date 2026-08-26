# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.math import exp
from std.bit import log2_ceil
from std.utils.numerics import max_finite, min_finite


comptime SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)
comptime GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
comptime BLOCK_DIM_X = 1 << log2_ceil(SIZE)


# ANCHOR: softmax_gpu_kernel
def softmax_gpu_kernel[
    input_size: Int,
    dtype: DType = .float32,
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
):
    comptime assert (
        dtype.is_floating_point()
    ), "dtype must be a floating-point type"

    var shared_max = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[input_size]())
    var shared_sum = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[input_size]())

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    # Load in values into shared_max, keeping the value in a variable for later
    var value: Scalar[dtype] = min_finite[dtype]()
    if global_i < input_size:
        value = input[global_i]
    shared_max[local_i] = value
    barrier()

    # Find max in log(n)
    var moffset = 1
    while moffset < input_size:
        if local_i + moffset < input_size:
            shared_max[local_i] = max(
                shared_max[local_i], shared_max[local_i + moffset]
            )
        barrier()
        moffset *= 2
    var m = shared_max[0]

    # Compute exp(value - max)
    var e: Scalar[dtype] = 0.0
    if global_i < input_size:
        e = exp(value - m)
    shared_sum[local_i] = e
    barrier()

    # Sum in log(n)
    var soffset = 1
    while soffset < input_size:
        if local_i + soffset < input_size:
            shared_sum[local_i] += shared_sum[local_i + soffset]
        barrier()
        soffset *= 2
    var sum = shared_sum[0]

    if global_i < input_size:
        output[global_i] = e / sum


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
def softmax_cpu_kernel[
    input_size: Int,
    dtype: DType = .float32,
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
):
    comptime assert (
        dtype.is_floating_point()
    ), "dtype must be a floating-point type"
    var m: Scalar[dtype] = min_finite[dtype]()
    for i in range(input_size):
        m = max(input[i], m)

    var sum: Scalar[dtype] = 0
    for i in range(input_size):
        var e = exp(input[i] - m)
        output[i] = e
        sum += e

    for i in range(input_size):
        output[i] /= sum


# ANCHOR_END: softmax_cpu_kernel

import extensibility

from extensibility import InputTensor, OutputTensor


@extensibility.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    def execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = .float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1, static_spec=_],
        input: InputTensor[dtype=dtype, rank=output.rank, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        var output_tensor = TileTensor[
            mut=True, dtype, LayoutType, MutAnyOrigin
        ](output.unsafe_ptr(), layout)
        var input_tensor = TileTensor[
            mut=True, dtype, LayoutType, MutAnyOrigin
        ](input.unsafe_ptr(), layout)

        comptime if target == "gpu":
            var gpu_ctx = ctx
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[dtype](
                    gpu_ctx,
                    output.unsafe_ptr(),
                    input_size,
                    owning=False,
                ),
                0,
            )

            comptime kernel = softmax_gpu_kernel[input_size, dtype]
            gpu_ctx.enqueue_function[kernel](
                output_tensor,
                input_tensor,
                grid_dim=1,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[input_size, dtype](output_tensor, input_tensor)
        else:
            raise Error("Unsupported target: " + target)
