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
import time
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


# ==============================================================================
# REFERENCE IMPLEMENTATIONS (NUMPY)
# ==============================================================================

def reference_attention_scaled(
    q: NDArray[np.float32], k: NDArray[np.float32], v: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Reference vector attention for arbitrary sequence lengths."""
    scores = np.dot(k, q)
    scores_max = np.max(scores)
    scores_exp = np.exp(scores - scores_max)
    attention_weights = scores_exp / np.sum(scores_exp)
    return np.dot(attention_weights, v)


def reference_attention_dynamic(
    q: NDArray[np.float32],
    k: NDArray[np.float32],
    v: NDArray[np.float32],
    actual_seq_len: int,
) -> NDArray[np.float32]:
    """Reference vector attention for dynamic sequence length (masking out-of-bound elements)."""
    # Only attend to valid rows 0..actual_seq_len-1
    k_valid = k[:actual_seq_len]
    v_valid = v[:actual_seq_len]
    scores = np.dot(k_valid, q)
    scores_max = np.max(scores)
    scores_exp = np.exp(scores - scores_max)
    attention_weights = scores_exp / np.sum(scores_exp)
    return np.dot(attention_weights, v_valid)


def reference_attention_batched(
    q: NDArray[np.float32], k: NDArray[np.float32], v: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Reference batched vector attention (Q: (batch_size, d), K: (seq_len, d), V: (seq_len, d))."""
    # Q @ K^T -> (batch_size, seq_len)
    scores = np.dot(q, k.T)
    # Row-wise softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
    # Weights @ V -> (batch_size, d)
    output = np.dot(weights, v)
    return output


# ==============================================================================
# MAX GRAPH EXECUTION HELPERS
# ==============================================================================

def get_attention_batched_model(
    batch_size: int,
    seq_len: int,
    d: int,
    session: InferenceSession,
    device: Device,
):
    """Build and compile MAX model for batched attention."""
    dtype = DType.float32
    mojo_kernels = Path(__file__).parent / "op"

    with Graph(
        "attention_batched_graph",
        input_types=[
            TensorType(dtype, shape=(batch_size, d), device=DeviceRef.from_device(device)),
            TensorType(dtype, shape=(seq_len, d), device=DeviceRef.from_device(device)),
            TensorType(dtype, shape=(seq_len, d), device=DeviceRef.from_device(device)),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        output = ops.custom(
            name="attention_batched",
            values=[graph.inputs[0], graph.inputs[1], graph.inputs[2]],
            device=DeviceRef.from_device(device),
            out_types=[
                TensorType(dtype=dtype, shape=(batch_size, d), device=DeviceRef.from_device(device))
            ],
            parameters={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "d": d,
                "dtype": dtype,
            },
        )[0].tensor
        graph.output(output)

    return session.load(graph)


def attention_scaled(
    q: NDArray[np.float32],
    k: NDArray[np.float32],
    v: NDArray[np.float32],
    session: InferenceSession,
    device: Device,
) -> Buffer:
    """Execute custom op 'attention_scaled' on target device."""
    dtype = DType.float32
    seq_len, d = k.shape

    q_tensor = Buffer.from_numpy(q).to(device)
    k_tensor = Buffer.from_numpy(k).to(device)
    v_tensor = Buffer.from_numpy(v).to(device)

    mojo_kernels = Path(__file__).parent / "op"

    with Graph(
        "attention_scaled_graph",
        input_types=[
            TensorType(dtype, shape=q_tensor.shape, device=DeviceRef.from_device(device)),
            TensorType(dtype, shape=k_tensor.shape, device=DeviceRef.from_device(device)),
            TensorType(dtype, shape=v_tensor.shape, device=DeviceRef.from_device(device)),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        output = ops.custom(
            name="attention_scaled",
            values=[graph.inputs[0], graph.inputs[1], graph.inputs[2]],
            device=DeviceRef.from_device(device),
            out_types=[TensorType(dtype=dtype, shape=(d,), device=DeviceRef.from_device(device))],
            parameters={"seq_len": seq_len, "d": d, "dtype": dtype},
        )[0].tensor
        graph.output(output)

    model = session.load(graph)
    result = model.execute(q_tensor, k_tensor, v_tensor)[0]
    assert isinstance(result, Buffer)
    return result.to(CPU()) if device == Accelerator() else result


def attention_dynamic(
    q: NDArray[np.float32],
    k: NDArray[np.float32],
    v: NDArray[np.float32],
    actual_seq_len: int,
    session: InferenceSession,
    device: Device,
) -> Buffer:
    """Execute custom op 'attention_dynamic' on target device."""
    dtype = DType.float32
    max_seq_len, d = k.shape

    q_tensor = Buffer.from_numpy(q).to(device)
    k_tensor = Buffer.from_numpy(k).to(device)
    v_tensor = Buffer.from_numpy(v).to(device)

    mojo_kernels = Path(__file__).parent / "op"

    with Graph(
        "attention_dynamic_graph",
        input_types=[
            TensorType(dtype, shape=q_tensor.shape, device=DeviceRef.from_device(device)),
            TensorType(dtype, shape=k_tensor.shape, device=DeviceRef.from_device(device)),
            TensorType(dtype, shape=v_tensor.shape, device=DeviceRef.from_device(device)),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        output = ops.custom(
            name="attention_dynamic",
            values=[graph.inputs[0], graph.inputs[1], graph.inputs[2]],
            device=DeviceRef.from_device(device),
            out_types=[TensorType(dtype=dtype, shape=(d,), device=DeviceRef.from_device(device))],
            parameters={
                "max_seq_len": max_seq_len,
                "actual_seq_len": actual_seq_len,
                "d": d,
                "dtype": dtype,
            },
        )[0].tensor
        graph.output(output)

    model = session.load(graph)
    result = model.execute(q_tensor, k_tensor, v_tensor)[0]
    assert isinstance(result, Buffer)
    return result.to(CPU()) if device == Accelerator() else result


def attention_batched(
    q: NDArray[np.float32],
    k: NDArray[np.float32],
    v: NDArray[np.float32],
    session: InferenceSession,
    device: Device,
) -> Buffer:
    """Execute custom op 'attention_batched' on target device."""
    batch_size, d = q.shape
    seq_len, _ = k.shape
    q_tensor = Buffer.from_numpy(q).to(device)
    k_tensor = Buffer.from_numpy(k).to(device)
    v_tensor = Buffer.from_numpy(v).to(device)

    model = get_attention_batched_model(batch_size, seq_len, d, session, device)
    result = model.execute(q_tensor, k_tensor, v_tensor)[0]
    assert isinstance(result, Buffer)
    return result.to(CPU()) if device == Accelerator() else result


# ==============================================================================
# TEST SUITE & STEP-BY-STEP VERIFICATION
# ==============================================================================

def test_challenge_1_1_scaled_sequence_length(gpu_session: InferenceSession):
    """
    Challenge 1.1: Sequence length scaling
    Test attention with SEQ_LEN = 32 and SEQ_LEN = 64.
    """
    print("\n" + "=" * 80)
    print("CHALLENGE 1.1: Sequence Length Scaling (SEQ_LEN = 32 & 64)")
    print("=" * 80)

    for seq_len, d in [(32, 16), (64, 32)]:
        print(f"\n---> Testing SEQ_LEN={seq_len}, D={d}")
        np.random.seed(42)
        q = np.random.randn(d).astype(np.float32) * 0.1
        k = np.random.randn(seq_len, d).astype(np.float32) * 0.1
        v = np.random.randn(seq_len, d).astype(np.float32) * 0.1

        expected = reference_attention_scaled(q, k, v)
        gpu_result = attention_scaled(q, k, v, gpu_session, Accelerator()).to_numpy()

        diff = np.max(np.abs(gpu_result - expected))
        print(f"  Output shape: {gpu_result.shape}")
        print(f"  Max absolute difference vs NumPy reference: {diff:.6e}")

        # TODO: [Challenge 1.1] SOLUTION NEEDED
        # Hint: Verify non-square matrix indexing in transpose_kernel (seq_len != d)
        # and ensure softmax_gpu_kernel_scaled handles log2_ceil(seq_len) threads correctly.
        try:
            np.testing.assert_allclose(
                gpu_result, expected, rtol=1e-4, atol=1e-4,
                err_msg=f"Challenge 1.1 failed for SEQ_LEN={seq_len}"
            )
            print(f"  ✓ PASSED for SEQ_LEN={seq_len}")
        except AssertionError as e:
            print(f"  ✗ FAILED for SEQ_LEN={seq_len}: {e.args[0].splitlines()[0] if e.args else e}")


def test_challenge_1_2_dynamic_sequence_length(gpu_session: InferenceSession):
    """
    Challenge 1.2: Dynamic sequence lengths
    Test attention with max_seq_len = 64 and actual_seq_len = 20 & 45.
    """
    print("\n" + "=" * 80)
    print("CHALLENGE 1.2: Dynamic Sequence Lengths (actual_seq_len <= max_seq_len)")
    print("=" * 80)

    max_seq_len = 64
    d = 16

    for actual_seq_len in [20, 45]:
        print(f"\n---> Testing max_seq_len={max_seq_len}, actual_seq_len={actual_seq_len}, D={d}")
        np.random.seed(42 + actual_seq_len)
        q = np.random.randn(d).astype(np.float32) * 0.1
        k = np.random.randn(max_seq_len, d).astype(np.float32) * 0.1
        v = np.random.randn(max_seq_len, d).astype(np.float32) * 0.1

        expected = reference_attention_dynamic(q, k, v, actual_seq_len)

        # TODO: [Challenge 1.2] SOLUTION NEEDED
        # Hint: In softmax_gpu_kernel_dynamic, set scores at global_i >= actual_seq_len
        # to min_finite[dtype]() (-infinity) before softmax reduction.
        gpu_result = attention_dynamic(q, k, v, actual_seq_len, gpu_session, Accelerator()).to_numpy()

        diff = np.max(np.abs(gpu_result - expected))
        print(f"  Max absolute difference vs masked NumPy reference: {diff:.6e}")
        try:
            np.testing.assert_allclose(
                gpu_result, expected, rtol=1e-4, atol=1e-4,
                err_msg=f"Challenge 1.2 failed for actual_seq_len={actual_seq_len}"
            )
            print(f"  ✓ PASSED for actual_seq_len={actual_seq_len}")
        except AssertionError as e:
            print(f"  ✗ FAILED for actual_seq_len={actual_seq_len}: {e.args[0].splitlines()[0] if e.args else e}")


def test_challenge_2_1_batched_attention(gpu_session: InferenceSession):
    """
    Challenge 2.1: Batched vector attention
    Test processing multiple query vectors Q(batch_size, d) simultaneously.
    """
    print("\n" + "=" * 80)
    print("CHALLENGE 2.1: Batched Vector Attention (Q shape: batch_size x d)")
    print("=" * 80)

    batch_size = 4
    seq_len = 32
    d = 16

    print(f"\n---> Testing batch_size={batch_size}, seq_len={seq_len}, D={d}")
    np.random.seed(123)
    q = np.random.randn(batch_size, d).astype(np.float32) * 0.1
    k = np.random.randn(seq_len, d).astype(np.float32) * 0.1
    v = np.random.randn(seq_len, d).astype(np.float32) * 0.1

    expected = reference_attention_batched(q, k, v)

    # TODO: [Challenge 2.1] SOLUTION NEEDED
    # Hint: Check rowwise_softmax_gpu_kernel and batched matmul indexing across batch dimension.
    gpu_result = attention_batched(q, k, v, gpu_session, Accelerator()).to_numpy()

    diff = np.max(np.abs(gpu_result - expected))
    print(f"  Output shape: {gpu_result.shape} (expected: ({batch_size}, {d}))")
    print(f"  Max absolute difference vs NumPy batched reference: {diff:.6e}")

    try:
        np.testing.assert_allclose(
            gpu_result, expected, rtol=1e-4, atol=1e-4,
            err_msg="Challenge 2.1 failed for batched attention"
        )
        print("  ✓ PASSED for batched attention")
    except AssertionError as e:
        print(f"  ✗ FAILED for batched attention: {e.args[0].splitlines()[0] if e.args else e}")


def test_challenge_2_2_batch_memory_optimization(gpu_session: InferenceSession):
    """
    Challenge 2.2: Memory optimization for batches
    Benchmark performance and compare execution times for batch sizes 2, 4, 8.
    """
    print("\n" + "=" * 80)
    print("CHALLENGE 2.2: Batch Performance & Memory Benchmarking (batch_size = 2, 4, 8)")
    print("=" * 80)

    seq_len = 32
    d = 16
    warmup_runs = 2
    timed_runs = 10

    for batch_size in [2, 4, 8]:
        np.random.seed(999 + batch_size)
        q = np.random.randn(batch_size, d).astype(np.float32) * 0.1
        k = np.random.randn(seq_len, d).astype(np.float32) * 0.1
        v = np.random.randn(seq_len, d).astype(np.float32) * 0.1

        q_tensor = Buffer.from_numpy(q).to(Accelerator())
        k_tensor = Buffer.from_numpy(k).to(Accelerator())
        v_tensor = Buffer.from_numpy(v).to(Accelerator())

        expected = reference_attention_batched(q, k, v)
        model = get_attention_batched_model(batch_size, seq_len, d, gpu_session, Accelerator())

        # Warmup
        for _ in range(warmup_runs):
            _ = model.execute(q_tensor, k_tensor, v_tensor)

        # Timing
        start_time = time.perf_counter()
        for _ in range(timed_runs):
            res = model.execute(q_tensor, k_tensor, v_tensor)[0]
        elapsed_ms = (time.perf_counter() - start_time) / timed_runs * 1000.0
        time_per_elem = elapsed_ms / batch_size

        assert isinstance(res, Buffer)
        gpu_arr = res.to(CPU()).to_numpy()
        matches = np.allclose(gpu_arr, expected, rtol=1e-4, atol=1e-4)

        status = "✓ VALID" if matches else "✗ INVALID"
        print(
            f"  Batch size {batch_size:2d}: Avg execution time = {elapsed_ms:.4f} ms "
            f"({time_per_elem:.4f} ms / query) [{status}]"
        )


if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print("PUZZLE 19 BONUS CHALLENGES TEST HARNESS")
    print(f"{'=' * 80}")

    gpu_session = InferenceSession(devices=[Accelerator()])

    # Run step-by-step tests for each bonus challenge
    test_challenge_1_1_scaled_sequence_length(gpu_session)
    test_challenge_1_2_dynamic_sequence_length(gpu_session)
    test_challenge_2_1_batched_attention(gpu_session)
    test_challenge_2_2_batch_memory_optimization(gpu_session)

    print(f"\n{'=' * 80}")
    print("ALL CHALLENGE VERIFICATIONS COMPLETED")
    print(f"{'=' * 80}\n")
