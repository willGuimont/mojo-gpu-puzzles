from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import TileTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import stack_allocation
from std.math import exp
from std.bit import log2_ceil
from std.utils.numerics import max_finite, min_finite
import extensibility

from extensibility import InputTensor, OutputTensor

from .attention import matmul_idiomatic_tiled, transpose_kernel


def softmax_gpu_kernel_scaled[
    input_size: Int,
    LayoutType: TensorLayout,
    dtype: DType = .float32,
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
):
    comptime assert (
        dtype.is_floating_point()
    ), "dtype must be a floating-point type"

    comptime block_dim_x = 1 << log2_ceil(input_size)
    comptime softmax_layout = row_major[block_dim_x]()

    var shared_max = stack_allocation[dtype=dtype, address_space=.SHARED](
        softmax_layout
    )
    var shared_sum = stack_allocation[dtype=dtype, address_space=.SHARED](
        softmax_layout
    )
    var global_i = thread_idx.x
    var input_lt = input.to_layout_tensor()
    var output_lt = output.to_layout_tensor()

    var val: Scalar[dtype] = min_finite[dtype]()
    if global_i < input_size:
        val = rebind[Scalar[dtype]](input_lt[global_i])
    shared_max[global_i] = val

    barrier()

    var stride = block_dim_x // 2
    while stride > 0:
        if global_i < stride:
            shared_max[global_i] = max(
                shared_max[global_i], shared_max[global_i + stride]
            )
        barrier()
        stride = stride // 2

    var block_max = shared_max[0]

    var exp_val: Scalar[dtype] = 0.0
    if global_i < input_size:
        exp_val = rebind[Scalar[dtype]](exp(val - block_max))
    shared_sum[global_i] = exp_val
    barrier()

    stride = block_dim_x // 2
    while stride > 0:
        if global_i < stride:
            shared_sum[global_i] += shared_sum[global_i + stride]
        barrier()
        stride = stride // 2

    var block_sum = shared_sum[0]

    if global_i < input_size:
        output_lt[global_i] = exp_val / block_sum


@extensibility.register("attention_scaled")
struct AttentionScaledCustomOp:
    @staticmethod
    def execute[
        target: StaticString,  # "cpu" or "gpu"
        seq_len: Int,
        d: Int,
        dtype: DType = .float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1, static_spec=_],  # (d,)
        q: InputTensor[dtype=dtype, rank=1, static_spec=_],  # (d,)
        k: InputTensor[dtype=dtype, rank=2, static_spec=_],  # (seq_len, d)
        v: InputTensor[dtype=dtype, rank=2, static_spec=_],  # (seq_len, d)
        ctx: DeviceContext,
    ) raises:
        comptime layout_q = row_major[d]()
        comptime layout_k = row_major[seq_len, d]()
        comptime layout_v = row_major[seq_len, d]()
        comptime layout_out = row_major[d]()

        comptime QLayout = type_of(layout_q)
        comptime KLayout = type_of(layout_k)
        comptime VLayout = type_of(layout_v)
        comptime OutLayout = type_of(layout_out)

        var output_tensor = TileTensor[
            mut=True, dtype, OutLayout, MutAnyOrigin
        ](output.unsafe_ptr(), layout_out)
        var q_tensor = TileTensor[mut=True, dtype, QLayout, MutAnyOrigin](
            q.unsafe_ptr(), layout_q
        )
        var k_tensor = TileTensor[mut=True, dtype, KLayout, MutAnyOrigin](
            k.unsafe_ptr(), layout_k
        )
        var v_tensor = TileTensor[mut=True, dtype, VLayout, MutAnyOrigin](
            v.unsafe_ptr(), layout_v
        )

        comptime if target == "gpu":
            # Challenge 1.1
            comptime layout_q_2d = row_major[1, d]()
            comptime Q2DLayout = type_of(layout_q_2d)
            comptime layout_k_t = row_major[d, seq_len]()
            comptime KTLayout = type_of(layout_k_t)
            comptime layout_scores_2d = row_major[1, seq_len]()
            comptime Scores2DLayout = type_of(layout_scores_2d)
            comptime layout_scores = row_major[seq_len]()
            comptime ScoresLayout = type_of(layout_scores)
            comptime layout_weights_2d = row_major[1, seq_len]()
            comptime Weights2DLayout = type_of(layout_weights_2d)
            comptime layout_result_2d = row_major[1, d]()
            comptime Result2DLayout = type_of(layout_result_2d)

            comptime TRANSPOSE_BLOCK_DIM_XY = 16
            comptime MATMUL_BLOCK_DIM_XY = 16

            comptime transpose_threads_per_block = (
                TRANSPOSE_BLOCK_DIM_XY,
                TRANSPOSE_BLOCK_DIM_XY,
            )
            comptime transpose_blocks_per_grid = (
                (d + TRANSPOSE_BLOCK_DIM_XY - 1) // TRANSPOSE_BLOCK_DIM_XY,
                (seq_len + TRANSPOSE_BLOCK_DIM_XY - 1)
                // TRANSPOSE_BLOCK_DIM_XY,
            )
            comptime matmul_threads_per_block = (
                MATMUL_BLOCK_DIM_XY,
                MATMUL_BLOCK_DIM_XY,
            )
            comptime scores_blocks_per_grid = (
                seq_len + MATMUL_BLOCK_DIM_XY - 1
            ) // MATMUL_BLOCK_DIM_XY
            comptime softmax_threads = 1 << log2_ceil(seq_len)
            comptime softmax_blocks_per_grid = 1
            comptime result_blocks_per_grid = (
                d + MATMUL_BLOCK_DIM_XY - 1
            ) // MATMUL_BLOCK_DIM_XY

            var k_t_buf = ctx.enqueue_create_buffer[dtype](seq_len * d)
            var scores_weights_buf = ctx.enqueue_create_buffer[dtype](seq_len)

            var k_t = TileTensor(k_t_buf, layout_k_t)
            var q_2d = q_tensor.reshape(layout_q_2d)

            comptime kernel = transpose_kernel[
                seq_len, d, KTLayout, KLayout, dtype
            ]
            ctx.enqueue_function[kernel](
                k_t,
                k_tensor,
                grid_dim=transpose_blocks_per_grid,
                block_dim=transpose_threads_per_block,
            )

            var score_2d = TileTensor(scores_weights_buf, layout_scores_2d)
            comptime kernel2 = matmul_idiomatic_tiled[
                1,
                seq_len,
                d,
                Scores2DLayout,
                Q2DLayout,
                KTLayout,
                dtype,
            ]
            ctx.enqueue_function[kernel2](
                score_2d,
                q_2d,
                k_t,
                grid_dim=scores_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

            var weights_out = TileTensor[
                mut=True, dtype, ScoresLayout, MutAnyOrigin
            ](scores_weights_buf, layout_scores)
            var weights_in = TileTensor[
                mut=True, dtype, ScoresLayout, MutAnyOrigin
            ](scores_weights_buf, layout_scores)
            comptime kernel3 = softmax_gpu_kernel_scaled[
                seq_len, ScoresLayout, dtype
            ]
            ctx.enqueue_function[kernel3](
                weights_out,
                weights_in,
                grid_dim=softmax_blocks_per_grid,
                block_dim=softmax_threads,
            )

            var weights_2d = score_2d.reshape(layout_weights_2d)
            var result_2d = output_tensor.reshape(layout_result_2d)
            comptime kernel4 = matmul_idiomatic_tiled[
                1,
                d,
                seq_len,
                Result2DLayout,
                Weights2DLayout,
                VLayout,
                dtype,
            ]
            ctx.enqueue_function[kernel4](
                result_2d,
                weights_2d,
                v_tensor,
                grid_dim=result_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

        elif target == "cpu":
            pass
        else:
            raise Error("Unsupported target: " + target)


def softmax_gpu_kernel_dynamic[
    max_seq_len: Int,
    LayoutType: TensorLayout,
    dtype: DType = .float32,
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    actual_seq_len_dev: Int32,
):
    comptime assert (
        dtype.is_floating_point()
    ), "dtype must be a floating-point type"

    comptime block_dim_x = 1 << log2_ceil(max_seq_len)
    comptime softmax_layout = row_major[block_dim_x]()

    var actual_seq_len = Int(actual_seq_len_dev)
    var shared_max = stack_allocation[dtype=dtype, address_space=.SHARED](
        softmax_layout
    )
    var shared_sum = stack_allocation[dtype=dtype, address_space=.SHARED](
        softmax_layout
    )
    var global_i = thread_idx.x
    var input_lt = input.to_layout_tensor()
    var output_lt = output.to_layout_tensor()

    # Challenge 1.2
    var val: Scalar[dtype] = min_finite[dtype]()
    if global_i < actual_seq_len:
        val = rebind[Scalar[dtype]](input_lt[global_i])
    shared_max[global_i] = val

    barrier()

    var stride = block_dim_x // 2
    while stride > 0:
        if global_i < stride:
            shared_max[global_i] = max(
                shared_max[global_i], shared_max[global_i + stride]
            )
        barrier()
        stride = stride // 2

    var block_max = shared_max[0]

    var exp_val: Scalar[dtype] = 0.0
    if global_i < actual_seq_len:
        exp_val = rebind[Scalar[dtype]](exp(val - block_max))
    shared_sum[global_i] = exp_val
    barrier()

    stride = block_dim_x // 2
    while stride > 0:
        if global_i < stride:
            shared_sum[global_i] += shared_sum[global_i + stride]
        barrier()
        stride = stride // 2

    var block_sum = shared_sum[0]

    if global_i < max_seq_len:
        output_lt[global_i] = exp_val / block_sum


@extensibility.register("attention_dynamic")
struct AttentionDynamicCustomOp:
    @staticmethod
    def execute[
        target: StaticString,
        max_seq_len: Int,
        actual_seq_len: Int,
        d: Int,
        dtype: DType = .float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1, static_spec=_],  # (d,)
        q: InputTensor[dtype=dtype, rank=1, static_spec=_],  # (d,)
        k: InputTensor[dtype=dtype, rank=2, static_spec=_],  # (max_seq_len, d)
        v: InputTensor[dtype=dtype, rank=2, static_spec=_],  # (max_seq_len, d)
        ctx: DeviceContext,
    ) raises:
        comptime layout_q = row_major[d]()
        comptime layout_k = row_major[max_seq_len, d]()
        comptime layout_v = row_major[max_seq_len, d]()
        comptime layout_out = row_major[d]()

        comptime QLayout = type_of(layout_q)
        comptime KLayout = type_of(layout_k)
        comptime VLayout = type_of(layout_v)
        comptime OutLayout = type_of(layout_out)

        var output_tensor = TileTensor[
            mut=True, dtype, OutLayout, MutAnyOrigin
        ](output.unsafe_ptr(), layout_out)
        var q_tensor = TileTensor[mut=True, dtype, QLayout, MutAnyOrigin](
            q.unsafe_ptr(), layout_q
        )
        var k_tensor = TileTensor[mut=True, dtype, KLayout, MutAnyOrigin](
            k.unsafe_ptr(), layout_k
        )
        var v_tensor = TileTensor[mut=True, dtype, VLayout, MutAnyOrigin](
            v.unsafe_ptr(), layout_v
        )

        comptime if target == "gpu":
            comptime layout_q_2d = row_major[1, d]()
            comptime Q2DLayout = type_of(layout_q_2d)
            comptime layout_k_t = row_major[d, max_seq_len]()
            comptime KTLayout = type_of(layout_k_t)
            comptime layout_scores_2d = row_major[1, max_seq_len]()
            comptime Scores2DLayout = type_of(layout_scores_2d)
            comptime layout_scores = row_major[max_seq_len]()
            comptime ScoresLayout = type_of(layout_scores)
            comptime layout_weights_2d = row_major[1, max_seq_len]()
            comptime Weights2DLayout = type_of(layout_weights_2d)
            comptime layout_result_2d = row_major[1, d]()
            comptime Result2DLayout = type_of(layout_result_2d)

            comptime TRANSPOSE_BLOCK_DIM_XY = 16
            comptime MATMUL_BLOCK_DIM_XY = 16

            comptime transpose_threads_per_block = (
                TRANSPOSE_BLOCK_DIM_XY,
                TRANSPOSE_BLOCK_DIM_XY,
            )
            comptime transpose_blocks_per_grid = (
                (d + TRANSPOSE_BLOCK_DIM_XY - 1) // TRANSPOSE_BLOCK_DIM_XY,
                (max_seq_len + TRANSPOSE_BLOCK_DIM_XY - 1)
                // TRANSPOSE_BLOCK_DIM_XY,
            )
            comptime matmul_threads_per_block = (
                MATMUL_BLOCK_DIM_XY,
                MATMUL_BLOCK_DIM_XY,
            )
            comptime scores_blocks_per_grid = (
                max_seq_len + MATMUL_BLOCK_DIM_XY - 1
            ) // MATMUL_BLOCK_DIM_XY
            comptime softmax_threads = 1 << log2_ceil(max_seq_len)
            comptime softmax_blocks_per_grid = 1
            comptime result_blocks_per_grid = (
                d + MATMUL_BLOCK_DIM_XY - 1
            ) // MATMUL_BLOCK_DIM_XY

            var k_t_buf = ctx.enqueue_create_buffer[dtype](max_seq_len * d)
            var scores_weights_buf = ctx.enqueue_create_buffer[dtype](
                max_seq_len
            )

            var k_t = TileTensor(k_t_buf, layout_k_t)
            var q_2d = q_tensor.reshape(layout_q_2d)

            comptime kernel = transpose_kernel[
                max_seq_len, d, KTLayout, KLayout, dtype
            ]
            ctx.enqueue_function[kernel](
                k_t,
                k_tensor,
                grid_dim=transpose_blocks_per_grid,
                block_dim=transpose_threads_per_block,
            )

            var score_2d = TileTensor(scores_weights_buf, layout_scores_2d)
            comptime kernel2 = matmul_idiomatic_tiled[
                1,
                max_seq_len,
                d,
                Scores2DLayout,
                Q2DLayout,
                KTLayout,
                dtype,
            ]
            ctx.enqueue_function[kernel2](
                score_2d,
                q_2d,
                k_t,
                grid_dim=scores_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

            var weights_out = TileTensor[
                mut=True, dtype, ScoresLayout, MutAnyOrigin
            ](scores_weights_buf, layout_scores)
            var weights_in = TileTensor[
                mut=True, dtype, ScoresLayout, MutAnyOrigin
            ](scores_weights_buf, layout_scores)
            comptime kernel3 = softmax_gpu_kernel_dynamic[
                max_seq_len, ScoresLayout, dtype
            ]
            ctx.enqueue_function[kernel3](
                weights_out,
                weights_in,
                Int32(actual_seq_len),
                grid_dim=softmax_blocks_per_grid,
                block_dim=softmax_threads,
            )

            var weights_2d = score_2d.reshape(layout_weights_2d)
            var result_2d = output_tensor.reshape(layout_result_2d)
            comptime kernel4 = matmul_idiomatic_tiled[
                1,
                d,
                max_seq_len,
                Result2DLayout,
                Weights2DLayout,
                VLayout,
                dtype,
            ]
            ctx.enqueue_function[kernel4](
                result_2d,
                weights_2d,
                v_tensor,
                grid_dim=result_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

        elif target == "cpu":
            pass
        else:
            raise Error("Unsupported target: " + target)


def rowwise_softmax_gpu_kernel[
    seq_len: Int,
    batch_size: Int,
    LayoutType: TensorLayout,
    dtype: DType = .float32,
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
):
    comptime assert (
        dtype.is_floating_point()
    ), "dtype must be a floating-point type"

    comptime block_dim_x = 1 << log2_ceil(seq_len)
    comptime softmax_layout = row_major[block_dim_x]()

    var row_idx = block_idx.y
    var col_idx = thread_idx.x

    var shared_max = stack_allocation[dtype=dtype, address_space=.SHARED](
        softmax_layout
    )
    var shared_sum = stack_allocation[dtype=dtype, address_space=.SHARED](
        softmax_layout
    )

    var input_lt = input.to_layout_tensor()
    var output_lt = output.to_layout_tensor()

    var val: Scalar[dtype] = min_finite[dtype]()
    if row_idx < batch_size and col_idx < seq_len:
        val = rebind[Scalar[dtype]](input_lt[row_idx, col_idx])
    shared_max[col_idx] = val

    barrier()

    var stride = block_dim_x // 2
    while stride > 0:
        if col_idx < stride:
            shared_max[col_idx] = max(
                shared_max[col_idx], shared_max[col_idx + stride]
            )
        barrier()
        stride = stride // 2

    var block_max = shared_max[0]

    var exp_val: Scalar[dtype] = 0.0
    if row_idx < batch_size and col_idx < seq_len:
        exp_val = rebind[Scalar[dtype]](exp(val - block_max))
    shared_sum[col_idx] = exp_val
    barrier()

    stride = block_dim_x // 2
    while stride > 0:
        if col_idx < stride:
            shared_sum[col_idx] += shared_sum[col_idx + stride]
        barrier()
        stride = stride // 2

    var block_sum = shared_sum[0]

    if row_idx < batch_size and col_idx < seq_len:
        output_lt[row_idx, col_idx] = exp_val / block_sum


@extensibility.register("attention_batched")
struct AttentionBatchedCustomOp:
    @staticmethod
    def execute[
        target: StaticString,
        batch_size: Int,
        seq_len: Int,
        d: Int,
        dtype: DType = .float32,
    ](
        output: OutputTensor[
            dtype=dtype, rank=2, static_spec=_
        ],  # (batch_size, d)
        q: InputTensor[dtype=dtype, rank=2, static_spec=_],  # (batch_size, d)
        k: InputTensor[dtype=dtype, rank=2, static_spec=_],  # (seq_len, d)
        v: InputTensor[dtype=dtype, rank=2, static_spec=_],  # (seq_len, d)
        ctx: DeviceContext,
    ) raises:
        comptime layout_q = row_major[batch_size, d]()
        comptime layout_k = row_major[seq_len, d]()
        comptime layout_v = row_major[seq_len, d]()
        comptime layout_out = row_major[batch_size, d]()

        comptime QLayout = type_of(layout_q)
        comptime KLayout = type_of(layout_k)
        comptime VLayout = type_of(layout_v)
        comptime OutLayout = type_of(layout_out)

        var output_tensor = TileTensor[
            mut=True, dtype, OutLayout, MutAnyOrigin
        ](output.unsafe_ptr(), layout_out)
        var q_tensor = TileTensor[mut=True, dtype, QLayout, MutAnyOrigin](
            q.unsafe_ptr(), layout_q
        )
        var k_tensor = TileTensor[mut=True, dtype, KLayout, MutAnyOrigin](
            k.unsafe_ptr(), layout_k
        )
        var v_tensor = TileTensor[mut=True, dtype, VLayout, MutAnyOrigin](
            v.unsafe_ptr(), layout_v
        )

        comptime if target == "gpu":
            comptime layout_k_t = row_major[d, seq_len]()
            comptime KTLayout = type_of(layout_k_t)
            comptime layout_scores_2d = row_major[batch_size, seq_len]()
            comptime Scores2DLayout = type_of(layout_scores_2d)

            comptime TRANSPOSE_BLOCK_DIM_XY = 16
            comptime MATMUL_BLOCK_DIM_XY = 16

            comptime transpose_threads_per_block = (
                TRANSPOSE_BLOCK_DIM_XY,
                TRANSPOSE_BLOCK_DIM_XY,
            )
            comptime transpose_blocks_per_grid = (
                (d + TRANSPOSE_BLOCK_DIM_XY - 1) // TRANSPOSE_BLOCK_DIM_XY,
                (seq_len + TRANSPOSE_BLOCK_DIM_XY - 1)
                // TRANSPOSE_BLOCK_DIM_XY,
            )

            comptime matmul_threads_per_block = (
                MATMUL_BLOCK_DIM_XY,
                MATMUL_BLOCK_DIM_XY,
            )
            comptime scores_blocks_per_grid = (
                (seq_len + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY,
                (batch_size + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY,
            )

            comptime softmax_threads = 1 << log2_ceil(seq_len)
            comptime softmax_blocks_per_grid = (1, batch_size)

            comptime result_blocks_per_grid = (
                (d + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY,
                (batch_size + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY,
            )

            var k_t_buf = ctx.enqueue_create_buffer[dtype](seq_len * d)
            var scores_weights_buf = ctx.enqueue_create_buffer[dtype](
                batch_size * seq_len
            )

            var k_t = TileTensor(k_t_buf, layout_k_t)
            var scores_tensor = TileTensor(scores_weights_buf, layout_scores_2d)

            # Step 1: Transpose K -> K^T
            comptime kernel = transpose_kernel[
                seq_len, d, KTLayout, KLayout, dtype
            ]
            ctx.enqueue_function[kernel](
                k_t,
                k_tensor,
                grid_dim=transpose_blocks_per_grid,
                block_dim=transpose_threads_per_block,
            )

            # Step 2: Compute batched scores Q @ K^T
            comptime kernel2 = matmul_idiomatic_tiled[
                batch_size,
                seq_len,
                d,
                Scores2DLayout,
                QLayout,
                KTLayout,
                dtype,
            ]
            ctx.enqueue_function[kernel2](
                scores_tensor,
                q_tensor,
                k_t,
                grid_dim=scores_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

            # Step 3: Row-wise Softmax over batched scores
            var scores_out = TileTensor[
                mut=True, dtype, Scores2DLayout, MutAnyOrigin
            ](scores_weights_buf, layout_scores_2d)
            var scores_in = TileTensor[
                mut=True, dtype, Scores2DLayout, MutAnyOrigin
            ](scores_weights_buf, layout_scores_2d)

            comptime kernel3 = rowwise_softmax_gpu_kernel[
                seq_len, batch_size, Scores2DLayout, dtype
            ]
            ctx.enqueue_function[kernel3](
                scores_out,
                scores_in,
                grid_dim=softmax_blocks_per_grid,
                block_dim=softmax_threads,
            )

            # Step 4: Compute final output Weights @ V
            comptime kernel4 = matmul_idiomatic_tiled[
                batch_size,
                d,
                seq_len,
                OutLayout,
                Scores2DLayout,
                VLayout,
                dtype,
            ]
            ctx.enqueue_function[kernel4](
                output_tensor,
                scores_tensor,
                v_tensor,
                grid_dim=result_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

        elif target == "cpu":
            pass
        else:
            raise Error("Unsupported target: " + target)
