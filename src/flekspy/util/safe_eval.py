import ast
import operator as op

import numpy as np


class SafeExpressionEvaluator:
    def __init__(self, context):
        self.context = context
        self._allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Attribute,
        }
        self._allowed_ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
        }
        self._allowed_functions = {
            "np": {
                "sqrt": np.sqrt,
                "log": np.log,
                "log10": np.log10,
                "abs": np.abs,
            }
        }

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
    return SafeExpressionEvaluator(context).eval(expression)
