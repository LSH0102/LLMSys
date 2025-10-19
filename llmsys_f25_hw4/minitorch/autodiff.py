from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """
        Accumulates the derivative (gradient) for this Variable.

        Args:
            x (Any): The gradient value to be accumulated.
        """
        pass

    @property
    def unique_id(self) -> int:
        """
        Returns:
            int: The unique identifier of this Variable.
        """
        pass

    def is_leaf(self) -> bool:
        """
        Returns whether this Variable is a leaf node in the computation graph.

        Returns:
            bool: True if this Variable is a leaf node, False otherwise.
        """
        pass

    def is_constant(self) -> bool:
        """
        Returns whether this Variable represents a constant value.

        Returns:
            bool: True if this Variable is constant, False otherwise.
        """
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """
        Returns the parent Variables of this Variable in the computation graph.

        Returns:
            Iterable[Variable]: The parent Variables of this Variable.
        """
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """
        Implements the chain rule to compute the gradient contributions of this Variable.

        Args:
            d_output (Any): The gradient of the output with respect to the Variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
                contains a parent Variable and the corresponding gradient contribution.
        """
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # BEGIN ASSIGN1_1
    # TODO
    
    visited=set()          #集合用来记录已经访问过的元素
    sorted_elements=[]            #用来储存已经排序好的元素
    stack=list([variable])       #构造一个stack数据结构
    
    
    while len(stack)>0:
        #抽出stack顶部的元素
        element=stack[-1]
        if element.is_constant():                                      #是常数就跳过
            visited.add(element.unique_id)
            stack.pop()
            continue
        if element.is_leaf()==True or element.unique_id in visited:  #如果是叶子结点或者已经访问完
            sorted_elements.append(element)
            stack.pop()
            if element.is_leaf()==True:
                visited.add(element.unique_id)
        else:                                              #对于既没有访问完且不是叶子节点的元素 访问它的前一个parent节点
            find=0
            for item in element.parents:
                if item.unique_id not in visited:
                    
                    parent_node=item
                    find=1
                    break
            
            if find==0:              #如果所有parent_node都被访问了 那说明这个节点已经被访问完 加入visited
                visited.add(element.unique_id)
            else:                          #否则访问这个parent_node
                stack.append(parent_node)
     
    
    return reversed(sorted_elements)          #输出的应该是从尾节点开始
            
            
    
    
    
    
    # END ASSIGN1_1


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # BEGIN ASSIGN1_1
    # TODO
    
    grads=dict()
    grads[variable.unique_id]=deriv
    
    topolist=topological_sort(variable)
    for node in topolist:
        if node.is_leaf()==True:
            node.accumulate_derivative(grads[node.unique_id])
        else:
            
            for item in node.chain_rule(grads[node.unique_id]):
                parent, grad=item[0],item[1]
                if parent.unique_id in grads:
                    grads[parent.unique_id]+=grad
                else:
                    grads[parent.unique_id]=grad
        
            
    
    # END ASSIGN1_1


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
