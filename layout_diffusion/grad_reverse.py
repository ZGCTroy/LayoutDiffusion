from torch.autograd import Function


# class GradReverse(Function):
#     def __init__(self, lambd):
#         self.lambd = lambd
#
#     def forward(self, x):
#         return x.view_as(x)
#
#     def backward(self, grad_output):
#         return (grad_output * -self.lambd)
#
#
# def grad_reverse(x, lambd=1.0):
#     return GradReverse(lambd)(x)


class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, grad_reverse_weight=1.0):
        ctx.grad_reverse_weight = grad_reverse_weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.grad_reverse_weight, None


def grad_reverse(x, grad_reverse_weight=1.0):
    return GradReverse.apply(x, grad_reverse_weight)
