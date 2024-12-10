# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT-compile a function for execution on a CUDA device.

    This function serves as a wrapper for Numba's `jit` decorator, specifically
    designed to compile functions that will run on a CUDA-enabled GPU. It allows
    for the optimization of performance by leveraging the parallel processing
    capabilities of CUDA.

    Args:
    ----
        fn (Fn): The function to be compiled for CUDA execution.
        **kwargs (Any): Additional parameters to customize the behavior of Numba's `jit`.

    Returns:
    -------
        Fn: The optimized compiled function that can be executed on a CUDA device.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT-compile a function for execution on a CUDA device.

    This function serves as a wrapper for Numba's `jit` decorator, specifically
    designed to compile functions that will run on a CUDA-enabled GPU. It allows
    for the optimization of performance by leveraging the parallel processing
    capabilities of CUDA.

    Args:
    ----
        fn (Fn): The function to be compiled for CUDA execution.
        **kwargs (Any): Additional parameters to customize the behavior of Numba's `jit`.

    Returns:
    -------
        FakeCUDAKernel: The optimized compiled function that can be executed on a CUDA device.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Apply a function element-wise to a tensor using CUDA.

        Args:
        ----
            fn (Callable[[float], float]): The function to apply to each element.

        Returns:
        -------
            MapProto: A function that applies `fn` to each element of a tensor.

        """
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Apply a function element-wise to two tensors using CUDA.

        Args:
        ----
            fn (Callable[[float, float], float]): The function to apply to each pair of elements.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that applies `fn` to each pair of elements of two tensors.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce a tensor along a specified dimension using a binary function.

        Args:
        ----
            fn (Callable[[float, float], float]): The binary function to apply for reduction.
            start (float, optional): The initial value for the reduction. Defaults to 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that reduces a tensor along the specified dimension.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 512
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication on two tensors using CUDA.

        Args:
        ----
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of the matrix multiplication.

        """
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i >= out_size:
            return

        to_index(i, out_shape, out_index)

        broadcast_index(out_index, out_shape, in_shape, in_index)

        in_pos = index_to_position(in_index, in_strides)
        out_pos = index_to_position(out_index, out_strides)

        out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i >= out_size:
            return

        to_index(i, out_shape, out_index)

        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)
        out_pos = index_to_position(out_index, out_strides)

        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    BLOCK_DIM = 32
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0
    cuda.syncthreads()
    step = 1
    while step < cuda.blockDim.x:
        if pos % (2 * step) == 0 and (pos + step) < cuda.blockDim.x:
            cache[pos] += cache[pos + step]
        cuda.syncthreads()
        step *= 2
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Compute the sum of elements in a tensor using a CUDA kernel.

    This function launches a CUDA kernel to perform a block-wise reduction
    on the input tensor `a`. The results are stored in a new tensor, which
    is allocated on the GPU. The function handles the necessary grid and
    block configurations for optimal performance.

    Args:
    ----
        a (Tensor): The input tensor whose elements are to be summed.

    Returns:
    -------
        TensorData: A tensor containing the sum of the elements in `a`.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if out_size > idx:
            to_index(idx, out_shape, out_index)
            output_position = index_to_position(out_index, out_strides)
            for i in range(a_shape[reduce_dim]):
                out_index[reduce_dim] = i
                in_pos = index_to_position(out_index, a_strides)
                reduce_value = fn(reduce_value, a_storage[in_pos])

            out[output_position] = reduce_value

    return jit(_reduce)


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Perform matrix multiplication on the GPU using shared memory for optimization.

    Args:
    ----
        out (Storage): The output storage where the result of the matrix multiplication will be stored.
        a (Storage): The first input matrix stored in a 1D array.
        b (Storage): The second input matrix stored in a 1D array.
        size (int): The size of the matrices (assuming square matrices).

    Returns:
    -------
        None

    """
    BLOCK_DIM = 32
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    row = by * cuda.blockDim.y + ty
    col = bx * cuda.blockDim.x + tx

    result = 0.0

    for t in range((size + BLOCK_DIM - 1) // BLOCK_DIM):
        if row < size and (t * BLOCK_DIM + tx) < size:
            shared_a[ty, tx] = a[row * size + (t * BLOCK_DIM + tx)]
        else:
            shared_a[ty, tx] = 0.0

        if col < size and (t * BLOCK_DIM + ty) < size:
            shared_b[ty, tx] = b[(t * BLOCK_DIM + ty) * size + col]
        else:
            shared_b[ty, tx] = 0.0

        cuda.syncthreads()

        for k in range(BLOCK_DIM):
            result += shared_a[ty, k] * shared_b[k, tx]

        cuda.syncthreads()

    if row < size and col < size:
        out[row * size + col] = result


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs matrix multiplication on two tensors using CUDA.

    Args:
    ----
        a (Tensor): The first input tensor, which should have shape (m, n).
        b (Tensor): The second input tensor, which should have shape (n, p).

    Returns:
    -------
        TensorData: A tensor containing the result of the matrix multiplication,
                     with shape (m, p).

    Note:
    ----
        The function assumes that the number of columns in the first tensor
        (a) matches the number of rows in the second tensor (b).

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    batch = cuda.blockIdx.z
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    row_index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col_index = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    thread_index_x = cuda.threadIdx.x
    thread_index_y = cuda.threadIdx.y
    accumulated_value = 0.0

    num_tiles = (a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM
    for tile_index in range(num_tiles):
        a_valid = (
            row_index < a_shape[-2]
            and tile_index * BLOCK_DIM + thread_index_y < a_shape[-1]
        )
        b_valid = (
            tile_index * BLOCK_DIM + thread_index_x < b_shape[-2]
            and col_index < b_shape[-1]
        )

        a_shared[thread_index_x, thread_index_y] = (
            a_storage[
                batch * a_batch_stride
                + row_index * a_strides[-2]
                + (tile_index * BLOCK_DIM + thread_index_y) * a_strides[-1]
            ]
            if a_valid
            else 0.0
        )

        b_shared[thread_index_x, thread_index_y] = (
            b_storage[
                batch * b_batch_stride
                + col_index * b_strides[-1]
                + (tile_index * BLOCK_DIM + thread_index_x) * b_strides[-2]
            ]
            if b_valid
            else 0.0
        )

        cuda.syncthreads()

        for k in range(BLOCK_DIM):
            accumulated_value += (
                a_shared[thread_index_x, k] * b_shared[k, thread_index_y]
            )

        cuda.syncthreads()

    if row_index < out_shape[-2] and col_index < out_shape[-1]:
        out_position = (
            batch * out_strides[0]
            + row_index * out_strides[1]
            + col_index * out_strides[2]
        )
        out[out_position] = accumulated_value


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
