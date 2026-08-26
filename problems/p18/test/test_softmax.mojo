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
from max.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from std.testing import assert_almost_equal
from std.bit import log2_ceil
from std.math import ceildiv, min, max, exp
from std.utils.numerics import min_finite

from op import (
    softmax_gpu_kernel,
    softmax_cpu_kernel,
    softmax_block_reduce_kernel,
    softmax_global_reduce_kernel,
    softmax_normalize_kernel,
)

comptime SIZE = 128
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)
comptime GRID_DIM_X = 1
comptime BLOCK_DIM_X = 1 << log2_ceil(SIZE)
comptime dtype = DType.float32

comptime SIZE_MULTI = 128
comptime BLOCK_DIM_MULTI = 32
comptime GRID_DIM_MULTI = ceildiv(SIZE_MULTI, BLOCK_DIM_MULTI)  # 4 blocks
comptime layout_multi = row_major[SIZE_MULTI]()
comptime LayoutTypeMulti = type_of(layout_multi)
comptime block_layout = row_major[GRID_DIM_MULTI]()
comptime BlockLayoutType = type_of(block_layout)
comptime stats_layout = row_major[2]()
comptime StatsLayoutType = type_of(stats_layout)


def test_softmax() raises:
    print("=== Running Single-Block Softmax Test ===")
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[.float32](SIZE)
        out.enqueue_fill(0)
        var inp = ctx.enqueue_create_buffer[.float32](SIZE)
        inp.enqueue_fill(0)
        # for CPU testing
        var expected = ctx.enqueue_create_host_buffer[.float32](SIZE)
        expected.enqueue_fill(0)
        var expected_tensor = TileTensor[
            mut=True, dtype, LayoutType, MutAnyOrigin
        ](expected, layout)

        # Initialize input and compute expected (CPU) inside map_to_host block
        with inp.map_to_host() as inp_host:
            for i in range(SIZE):
                inp_host[i] = Scalar[dtype](i)

            print("Input values:")
            for i in range(SIZE):
                print(inp_host[i], end=" ")
            print()
            # Create layout tensor for CPU calculation (must stay inside with block)
            var input_host_tensor = TileTensor[
                mut=True, dtype, LayoutType, MutAnyOrigin
            ](inp_host, layout)
            # Compute expected results using our CPU kernel while inp_host is valid
            softmax_cpu_kernel[SIZE, dtype](expected_tensor, input_host_tensor)

        # for GPU testing
        var output_tensor = TileTensor(out, layout)
        var input_tensor = TileTensor[
            mut=True, dtype, LayoutType, MutAnyOrigin
        ](inp, layout)

        # Run GPU kernel
        comptime kernel = softmax_gpu_kernel[SIZE, dtype]
        ctx.enqueue_function[kernel](
            output_tensor,
            input_tensor,
            grid_dim=GRID_DIM_X,
            block_dim=BLOCK_DIM_X,
        )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("GPU softmax results:")
            for i in range(SIZE):
                print(out_host[i], end=" ")
            print()

            print("Expected results:")
            for i in range(SIZE):
                print(expected[i], end=" ")
            print()

            var sum_gpu: Float32 = 0.0
            for i in range(SIZE):
                sum_gpu += out_host[i]
                assert_almost_equal(
                    out_host[i], expected[i], atol=1e-5, rtol=1e-5
                )

            print("Sum of probabilities:", sum_gpu)
            assert_almost_equal(sum_gpu, 1.0, atol=1e-5, rtol=1e-5)
            print("Single-block test passed 🎉\n")


def test_softmax_block_reduce() raises:
    print("=== Testing Pass 1: softmax_block_reduce_kernel ===")
    with DeviceContext() as ctx:
        var inp = ctx.enqueue_create_buffer[dtype](SIZE_MULTI)
        var block_max_buf = ctx.enqueue_create_buffer[dtype](GRID_DIM_MULTI)
        var block_sum_buf = ctx.enqueue_create_buffer[dtype](GRID_DIM_MULTI)

        with inp.map_to_host() as inp_host:
            for i in range(SIZE_MULTI):
                inp_host[i] = Scalar[dtype](i)

        var input_tensor = TileTensor[
            mut=True, dtype, LayoutTypeMulti, MutAnyOrigin
        ](inp, layout_multi)
        var block_max_tensor = TileTensor[
            mut=True, dtype, BlockLayoutType, MutAnyOrigin
        ](block_max_buf, block_layout)
        var block_sum_tensor = TileTensor[
            mut=True, dtype, BlockLayoutType, MutAnyOrigin
        ](block_sum_buf, block_layout)

        comptime reduce_blocks_kernel = softmax_block_reduce_kernel[
            SIZE_MULTI, BLOCK_DIM_MULTI, dtype
        ]
        ctx.enqueue_function[reduce_blocks_kernel](
            input_tensor,
            block_max_tensor,
            block_sum_tensor,
            grid_dim=GRID_DIM_MULTI,
            block_dim=BLOCK_DIM_MULTI,
        )
        ctx.synchronize()

        # Compute expected block_max and block_sum on CPU
        with inp.map_to_host() as inp_host, block_max_buf.map_to_host() as b_max_host, block_sum_buf.map_to_host() as b_sum_host:
            for b in range(GRID_DIM_MULTI):
                var expected_max: Float32 = min_finite[dtype]()
                for i in range(BLOCK_DIM_MULTI):
                    var idx = b * BLOCK_DIM_MULTI + i
                    if idx < SIZE_MULTI:
                        expected_max = max(expected_max, inp_host[idx])

                var expected_sum: Float32 = 0.0
                for i in range(BLOCK_DIM_MULTI):
                    var idx = b * BLOCK_DIM_MULTI + i
                    if idx < SIZE_MULTI:
                        expected_sum += exp(inp_host[idx] - expected_max)

                print("Block", b, "GPU Max:", b_max_host[b], "Expected Max:", expected_max)
                print("Block", b, "GPU Sum:", b_sum_host[b], "Expected Sum:", expected_sum)
                assert_almost_equal(b_max_host[b], expected_max, atol=1e-5, rtol=1e-5)
                assert_almost_equal(b_sum_host[b], expected_sum, atol=1e-5, rtol=1e-5)
        print("Pass 1 (softmax_block_reduce_kernel) Passed! 🎉\n")


def test_softmax_global_reduce() raises:
    print("=== Testing Pass 2: softmax_global_reduce_kernel ===")
    with DeviceContext() as ctx:
        var block_max_buf = ctx.enqueue_create_buffer[dtype](GRID_DIM_MULTI)
        var block_sum_buf = ctx.enqueue_create_buffer[dtype](GRID_DIM_MULTI)
        var global_stats_buf = ctx.enqueue_create_buffer[dtype](2)

        # Set up known block stats on CPU
        with block_max_buf.map_to_host() as b_max_host, block_sum_buf.map_to_host() as b_sum_host:
            for b in range(GRID_DIM_MULTI):
                b_max_host[b] = Scalar[dtype]((b + 1) * 10)
                b_sum_host[b] = Scalar[dtype](2.0)

        var block_max_tensor = TileTensor[
            mut=True, dtype, BlockLayoutType, MutAnyOrigin
        ](block_max_buf, block_layout)
        var block_sum_tensor = TileTensor[
            mut=True, dtype, BlockLayoutType, MutAnyOrigin
        ](block_sum_buf, block_layout)
        var global_stats_tensor = TileTensor[
            mut=True, dtype, StatsLayoutType, MutAnyOrigin
        ](global_stats_buf, stats_layout)

        comptime reduce_global_kernel = softmax_global_reduce_kernel[
            GRID_DIM_MULTI, dtype
        ]
        ctx.enqueue_function[reduce_global_kernel](
            block_max_tensor,
            block_sum_tensor,
            global_stats_tensor,
            grid_dim=1,
            block_dim=min(GRID_DIM_MULTI, BLOCK_DIM_MULTI),
        )
        ctx.synchronize()

        # Calculate expected global max and global sum on CPU
        var expected_global_max: Float32 = min_finite[dtype]()
        with block_max_buf.map_to_host() as b_max_host:
            for b in range(GRID_DIM_MULTI):
                expected_global_max = max(expected_global_max, b_max_host[b])

        var expected_global_sum: Float32 = 0.0
        with block_max_buf.map_to_host() as b_max_host, block_sum_buf.map_to_host() as b_sum_host:
            for b in range(GRID_DIM_MULTI):
                expected_global_sum += b_sum_host[b] * exp(b_max_host[b] - expected_global_max)

        with global_stats_buf.map_to_host() as stats_host:
            print("GPU Global Max:", stats_host[0], "Expected:", expected_global_max)
            print("GPU Global Sum:", stats_host[1], "Expected:", expected_global_sum)
            assert_almost_equal(stats_host[0], expected_global_max, atol=1e-5, rtol=1e-5)
            assert_almost_equal(stats_host[1], expected_global_sum, atol=1e-5, rtol=1e-5)
        print("Pass 2 (softmax_global_reduce_kernel) Passed! 🎉\n")


def test_softmax_normalize() raises:
    print("=== Testing Pass 3: softmax_normalize_kernel ===")
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE_MULTI)
        var inp = ctx.enqueue_create_buffer[dtype](SIZE_MULTI)
        var global_stats_buf = ctx.enqueue_create_buffer[dtype](2)

        var g_max: Float32 = 127.0
        var g_sum: Float32 = 1.7182818
        with global_stats_buf.map_to_host() as stats_host:
            stats_host[0] = Scalar[dtype](g_max)
            stats_host[1] = Scalar[dtype](g_sum)

        with inp.map_to_host() as inp_host:
            for i in range(SIZE_MULTI):
                inp_host[i] = Scalar[dtype](i)

        var output_tensor = TileTensor(out, layout_multi)
        var input_tensor = TileTensor[
            mut=True, dtype, LayoutTypeMulti, MutAnyOrigin
        ](inp, layout_multi)
        var global_stats_tensor = TileTensor[
            mut=True, dtype, StatsLayoutType, MutAnyOrigin
        ](global_stats_buf, stats_layout)

        comptime normalize_kernel = softmax_normalize_kernel[SIZE_MULTI, dtype]
        ctx.enqueue_function[normalize_kernel](
            output_tensor,
            input_tensor,
            global_stats_tensor,
            grid_dim=GRID_DIM_MULTI,
            block_dim=BLOCK_DIM_MULTI,
        )
        ctx.synchronize()

        with inp.map_to_host() as inp_host, out.map_to_host() as out_host:
            for i in range(SIZE_MULTI):
                var expected = exp(inp_host[i] - g_max) / g_sum
                assert_almost_equal(out_host[i], expected, atol=1e-5, rtol=1e-5)
        print("Pass 3 (softmax_normalize_kernel) Passed! 🎉\n")


def test_softmax_challenge() raises:
    print("=== Running Multi-Block Buffer Reduction Softmax Integration Test ===")
    with DeviceContext() as ctx:
        # Primary buffers
        var out = ctx.enqueue_create_buffer[dtype](SIZE_MULTI)
        out.enqueue_fill(0)
        var inp = ctx.enqueue_create_buffer[dtype](SIZE_MULTI)
        inp.enqueue_fill(0)

        # Hierarchical reduction buffers
        var block_max_buf = ctx.enqueue_create_buffer[dtype](GRID_DIM_MULTI)
        var block_sum_buf = ctx.enqueue_create_buffer[dtype](GRID_DIM_MULTI)
        var global_stats_buf = ctx.enqueue_create_buffer[dtype](
            2
        )  # [0]: max, [1]: sum
        block_max_buf.enqueue_fill(0)
        block_sum_buf.enqueue_fill(0)
        global_stats_buf.enqueue_fill(0)

        # CPU verification buffer
        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE_MULTI)
        expected.enqueue_fill(0)
        var expected_tensor = TileTensor[
            mut=True, dtype, LayoutTypeMulti, MutAnyOrigin
        ](expected, layout_multi)

        # Initialize input and compute CPU baseline
        with inp.map_to_host() as inp_host:
            for i in range(SIZE_MULTI):
                inp_host[i] = Scalar[dtype](i)

            var input_host_tensor = TileTensor[
                mut=True, dtype, LayoutTypeMulti, MutAnyOrigin
            ](inp_host, layout_multi)
            softmax_cpu_kernel[SIZE_MULTI, dtype](
                expected_tensor, input_host_tensor
            )

        # Device tensor views
        var output_tensor = TileTensor(out, layout_multi)
        var input_tensor = TileTensor[
            mut=True, dtype, LayoutTypeMulti, MutAnyOrigin
        ](inp, layout_multi)
        var block_max_tensor = TileTensor[
            mut=True, dtype, BlockLayoutType, MutAnyOrigin
        ](block_max_buf, block_layout)
        var block_sum_tensor = TileTensor[
            mut=True, dtype, BlockLayoutType, MutAnyOrigin
        ](block_sum_buf, block_layout)
        var global_stats_tensor = TileTensor[
            mut=True, dtype, StatsLayoutType, MutAnyOrigin
        ](global_stats_buf, stats_layout)

        # Pass 1: Block-level local reduction
        comptime reduce_blocks_kernel = softmax_block_reduce_kernel[
            SIZE_MULTI, BLOCK_DIM_MULTI, dtype
        ]
        ctx.enqueue_function[reduce_blocks_kernel](
            input_tensor,
            block_max_tensor,
            block_sum_tensor,
            grid_dim=GRID_DIM_MULTI,
            block_dim=BLOCK_DIM_MULTI,
        )

        # Pass 2: Global reduction aggregation
        comptime reduce_global_kernel = softmax_global_reduce_kernel[
            GRID_DIM_MULTI, dtype
        ]
        ctx.enqueue_function[reduce_global_kernel](
            block_max_tensor,
            block_sum_tensor,
            global_stats_tensor,
            grid_dim=1,
            block_dim=min(GRID_DIM_MULTI, BLOCK_DIM_MULTI),
        )

        # Pass 3: Elementwise normalization
        comptime normalize_kernel = softmax_normalize_kernel[SIZE_MULTI, dtype]
        ctx.enqueue_function[normalize_kernel](
            output_tensor,
            input_tensor,
            global_stats_tensor,
            grid_dim=GRID_DIM_MULTI,
            block_dim=BLOCK_DIM_MULTI,
        )

        ctx.synchronize()

        # Validation
        with out.map_to_host() as out_host:
            var sum_gpu: Float32 = 0.0
            for i in range(SIZE_MULTI):
                sum_gpu += out_host[i]
                assert_almost_equal(
                    out_host[i], expected[i], atol=1e-5, rtol=1e-5
                )

            print("Sum of probabilities:", sum_gpu)
            assert_almost_equal(sum_gpu, 1.0, atol=1e-5, rtol=1e-5)
            print("Multi-block test passed 🎉\n")


def main() raises:
    test_softmax()
    test_softmax_block_reduce()
    test_softmax_global_reduce()
    test_softmax_normalize()
    test_softmax_challenge()
