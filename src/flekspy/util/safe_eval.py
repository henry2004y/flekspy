import ast
import operator as op

import numpy as np


class SafeExpressionEvaluator:
    """A safe evaluator for Python expressions using an Abstract Syntax Tree (AST).

    This class parses a string expression, validates it against a whitelist of
    allowed AST nodes, operators, and functions, and then evaluates it.
    This provides a secure alternative to `eval()` for handling simple
    mathematical expressions from untrusted sources.
    """
    _allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Constant,
    }
    _allowed_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }
    _allowed_functions = {
        "np": {
            "sqrt": np.sqrt,
            "log": np.log,
            "log10": np.log10,
            "abs": np.abs,
        }
    }

    def __init__(self, context):
        self.context = context

    def _eval_node(self, node):
        if not isinstance(node, ast.AST):
            raise TypeError(f"Unsupported node type: {type(node)}")

        node_type = type(node)
        if node_type not in self._allowed_nodes:
            raise ValueError(f"Unsupported node type: {node_type.__name__}")

        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in self.context:
                return self.context[node.id]
            raise NameError(f"Name '{node.id}' is not defined in the context.")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type in self._allowed_ops:
                return self._allowed_ops[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in self._allowed_ops:
                return self._allowed_ops[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                namespace = node.func.value.id
                func_name = node.func.attr
                if (
                    namespace in self._allowed_functions
                    and func_name in self._allowed_functions[namespace]
                ):
                    args = [self._eval_node(arg) for arg in node.args]
                    return self._allowed_functions[namespace][func_name](*args)
            raise NameError(
                f"Unsupported function call: {ast.dump(node.func)}"
            )
        else:
            raise TypeError(f"Unsupported node type: {node_type.__name__}")

    def eval(self, expression):
        try:
            node = ast.parse(expression, mode="eval")
            return self._eval_node(node)
        except (ValueError, TypeError, NameError, SyntaxError) as e:
            raise type(e)(f"Failed to evaluate expression: {e}") from e


def safe_eval(expression, context):
    """Safely evaluates a string expression in a given context.

    Args:
        expression (str): The expression to evaluate.
        context (dict): A dictionary of allowed variable names and their values.

    Returns:
        The result of the evaluated expression.

    Raises:
        ValueError: If the expression contains unsupported nodes, operators, or functions.
        NameError: If the expression contains undefined variables or unsupported function calls.
        SyntaxError: If the expression is not valid Python syntax.
        TypeError: If an operation is applied to an object of an inappropriate type.
    """
    return SafeExpressionEvaluator(context).eval(expression)
