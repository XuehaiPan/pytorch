import torch
import torch.utils._cxx_pytree as pytree


@torch.compile(fullgraph=True)
def func(xs):
    leaves = pytree.tree_leaves(xs)
    return sum(leaves)


x = torch.randn(3, 2)
tree = {
    'a': [x, x - 1],
    'b': x + 1,
    'c': (x,),
}

y = func(tree)
