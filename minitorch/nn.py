from typing import Tuple

from .tensor import Tensor
from .tensor_functions import rand
from typing import Optional
from .tensor_functions import Function, tensor

from .autodiff import Context


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor (Go back to argmax)
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # Calculate new dimensions after tiling
    new_height = height // kh
    new_width = width // kw

    # Reshape input into tiles
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Rearrange dimensions to group kernel dimensions at the end
    tiled = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Combine kernel dimensions into single dimension
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling using the specified kernel.

    Args:
    ----
        input (Tensor): The input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): The height and width of the pooling kernel.

    Returns:
    -------
        Tensor: The pooled tensor of shape (batch, channel, new_height, new_width).

    """
    tiled_tensor, output_height, output_width = tile(input, kernel)
    averaged_tensor = tiled_tensor.mean(dim=4)
    return averaged_tensor.view(
        input.shape[0], input.shape[1], output_height, output_width
    )


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes the forward pass of max reduction.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): Input tensor to reduce.
            dim (Tensor): The dimension to reduce over.

        Returns:
        -------
            Tensor: The maximum values along the specified dimension.

        """
        ctx.save_for_backward(a, dim)
        return a.f.max_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass of max reduction.

        Args:
        ----
            ctx (Context): The context containing saved tensors from forward pass.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Tuple containing:
                - Gradient for input tensor (grad_output * one_hot mask)
                - Zero gradient for the dimension parameter

        """
        a, dim = ctx.saved_values
        one_hot = argmax(a, int(dim.item()))
        return grad_output * one_hot, tensor([0.0])


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Apply max reduction along the specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (Optional[int]): The dimension to reduce over. If None, reduces over all dimensions.

    Returns:
    -------
        Tensor: The maximum values along the specified dimension.

    """
    dim_tensor = input._ensure_tensor(0 if dim is None else dim)
    input_tensor = input.contiguous().view(input.size) if dim is None else input
    return Max.apply(input_tensor, dim_tensor)


def argmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (Optional[int]): The dimension to compute the argmax over.

    Returns:
    -------
        Tensor: A tensor with 1-hot encoding of the argmax positions.

    """
    dim_tensor = input._ensure_tensor(0 if dim is None else dim)
    max_values = input.f.max_reduce(input, int(dim_tensor.item()))
    return max_values == input


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax along the specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension to compute the softmax over.

    Returns:
    -------
        Tensor: The softmax values along the specified dimension.

    """
    shifted_input = input - max(input, dim)
    exp_shifted = shifted_input.exp()
    normalization = exp_shifted.sum(dim)
    return exp_shifted / normalization


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax along the specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension to compute the log softmax over.

    Returns:
    -------
        Tensor: The log softmax values along the specified dimension.

    """
    max_val = max(input, dim)

    shifted_input = input - max_val
    exp_shifted = shifted_input.exp()
    log_sum = exp_shifted.sum(dim).log()
    # Final result is x - max_val - log(sum(exp(x - max_val)))
    return shifted_input - log_sum


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling using the specified kernel.

    Args:
    ----
        input (Tensor): The input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): The height and width of the pooling kernel.

    Returns:
    -------
        Tensor: The pooled tensor of shape (batch, channel, new_height, new_width).

    """
    batch_size, channels, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    pooled = max(tiled, dim=4)
    return pooled.view(batch_size, channels, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor.

    Args:
    ----
        input (Tensor): The input tensor.
        rate (float): The dropout rate (probability of dropping a unit).
        ignore (bool): If True, dropout is not applied.

    Returns:
    -------
        Tensor: The tensor after applying dropout.

    """
    if ignore:
        return input
    if rate >= 1.0:
        return input * 0
    if rate <= 0.0:
        return input
    mask = rand(input.shape) > rate
    return input * mask
