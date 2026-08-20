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
from max.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.sys import argv
from std.math import log2
from std.testing import assert_equal

# ANCHOR: prefix_sum_simple
comptime TPB = 8
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)


def prefix_sum_simple(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    var shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB]())

    if local_i < size:
        shared[local_i] = a[global_i]

    barrier()

    # 1| 2, 3, 4, 5
    # Add i - 1, except for i < 1
    # 1, 3| 5, 7, 9
    # Add i - 2, except for i < 2
    # 1, 3, 6, 10| 14
    # Add i - 4, except for i < 4
    # 1, 3, 6, 10, 15|

    var delta = 1
    while delta < size:
        var sum = shared[local_i]
        if local_i >= delta and local_i < size:
            sum += shared[local_i - delta]

        barrier()
        shared[local_i] = sum
        barrier()
        delta *= 2

    if global_i < size:
        output[global_i] = shared[local_i]


# ANCHOR_END: prefix_sum_simple

comptime SIZE_2 = 15
comptime BLOCKS_PER_GRID_2 = (2, 1)
comptime THREADS_PER_BLOCK_2 = (TPB, 1)
comptime EXTENDED_SIZE = SIZE_2 + 2  # up to 2 blocks
comptime layout_2 = row_major[SIZE_2]()
comptime extended_layout = row_major[EXTENDED_SIZE]()
comptime Layout2Type = type_of(layout_2)
comptime ExtendedLayout = type_of(extended_layout)

# ANCHOR: prefix_sum_complete


# Kernel 1: Compute local prefix sums and store block sums in out
def prefix_sum_local_phase(
    output: TileTensor[mut=True, dtype, ExtendedLayout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, Layout2Type, ImmutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    var shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB]())

    if local_i < size:
        shared[local_i] = a[global_i]

    barrier()

    var delta = 1
    while delta < size:
        var sum = shared[local_i]
        if local_i >= delta and local_i < size:
            sum += shared[local_i - delta]

        barrier()
        shared[local_i] = sum
        barrier()
        delta *= 2

    if global_i < size:
        output[global_i] = shared[local_i]


# Kernel 2: Add block sums to their respective blocks
def prefix_sum_block_sum_phase(
    output: TileTensor[mut=True, dtype, ExtendedLayout, MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    
    # if block_idx.x > 0 and global_i < size:
    #     # Assumes only two blocks
    #     var prev_sum = output[size + block_idx.x - 1]
    #     output[global_i] += prev_sum


# ANCHOR_END: prefix_sum_complete


def main() raises:
    with DeviceContext() as ctx:
        var use_simple = argv()[1] == "--simple"
        var size = SIZE if use_simple else SIZE_2
        var num_blocks = (size + TPB - 1) // TPB

        if not use_simple and num_blocks > EXTENDED_SIZE - SIZE_2:
            raise Error("Extended buffer too small for the number of blocks")

        var buffer_size = size if use_simple else EXTENDED_SIZE
        var out = ctx.enqueue_create_buffer[dtype](buffer_size)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](size)
        a.enqueue_fill(0)

        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = Scalar[dtype](i)

        if use_simple:
            var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)
            var out_tensor = TileTensor(out, layout)

            ctx.enqueue_function[prefix_sum_simple](
                out_tensor,
                a_tensor,
                Int32(size),
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        else:
            var a_tensor = TileTensor[mut=False, dtype, Layout2Type](
                a, layout_2
            )
            var out_tensor = TileTensor(out, extended_layout)

            # ANCHOR: prefix_sum_complete_block_level_sync
            # Phase 1: Local prefix sums
            ctx.enqueue_function[prefix_sum_local_phase](
                out_tensor,
                a_tensor,
                Int32(size),
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )

            # Phase 2: Add block sums
            ctx.enqueue_function[prefix_sum_block_sum_phase](
                out_tensor,
                Int32(size),
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )
            # ANCHOR_END: prefix_sum_complete_block_level_sync

        # Verify results for both cases
        var expected = ctx.enqueue_create_host_buffer[dtype](size)
        expected.enqueue_fill(0)
        ctx.synchronize()

        with a.map_to_host() as a_host:
            expected[0] = a_host[0]
            for i in range(1, size):
                expected[i] = expected[i - 1] + a_host[i]

        with out.map_to_host() as out_host:
            if not use_simple:
                print(
                    "Note: we print the extended buffer here, but we only need"
                    " to print the first `size` elements"
                )

            print("out:", out_host)
            print("expected:", expected)
            # Here we need to use the size of the original array, not the extended one
            size = size if use_simple else SIZE_2
            for i in range(size):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 14 complete ✅")
