# Personal Solution to Homeworks of CMU11868：LLMSYS
My personal homework solutions.

This project repository is for learning and communication purposes only.

The code implementation is not guaranteed to be correct.
## Updates:
### HW1 
Uploaded on 2025.10.14

Passed all tests
### HW2
Uploaded on 2025.10.15

Passed all tests
### HW3
Uploaded on 2025.10.18

Passed all tests
modified the View function in minitorch/tensor_functions.py

```python
@staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        if grad_output._tensor.is_contiguous()==False:
            grad_output=grad_output.contiguous()
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )
Added two lines to guarantee that the grad is contiguous, other the grad would not propagate properly due to the use of permute.
### HW4
Completed on 2025.10.18

Passed all tests
