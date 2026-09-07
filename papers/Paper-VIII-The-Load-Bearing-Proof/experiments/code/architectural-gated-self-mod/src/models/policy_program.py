from __future__ import annotations

import ast
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, Callable

ALLOWED_NAMES = {
    "loss", "loss_delta", "grad_norm", "param_norm", "retention", "task_progress", "improvement", "bias",
    "lr_scale", "grad_clip", "noise_scale",
    "sigmoid", "tanh", "relu", "softplus", "clip", "min", "max", "abs", "float",
}
ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.Expr, ast.Load, ast.Store, ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.Call, ast.Compare,
    ast.IfExp, ast.keyword, ast.Subscript, ast.Dict, ast.Tuple,
)
ACTIVATIONS = ["sigmoid", "tanh", "relu", "softplus"]
FEATURES = ["loss", "loss_delta", "grad_norm", "param_norm", "retention", "task_progress", "improvement"]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def tanh(x: float) -> float:
    return math.tanh(x)


def relu(x: float) -> float:
    return max(0.0, x)


def softplus(x: float) -> float:
    if x > 20:
        return x
    return math.log1p(math.exp(x))


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class PolicyGenome:
    lr_feature: str = "retention"
    clip_feature: str = "grad_norm"
    noise_feature: str = "loss_delta"
    lr_activation: str = "sigmoid"
    clip_activation: str = "softplus"
    noise_activation: str = "sigmoid"
    lr_w: float = 0.8
    lr_b: float = 0.1
    clip_w: float = 0.5
    clip_b: float = 1.0
    noise_w: float = -0.5
    noise_b: float = -2.5

    def render_source(self) -> str:
        return f'''
def policy(features):
    loss = float(features["loss"])
    loss_delta = float(features["loss_delta"])
    grad_norm = float(features["grad_norm"])
    param_norm = float(features["param_norm"])
    retention = float(features["retention"])
    task_progress = float(features["task_progress"])
    improvement = float(features["improvement"])
    bias = 1.0

    lr_scale = clip({self.lr_activation}({self.lr_w:.6f}*{self.lr_feature} + {self.lr_b:.6f}) * 2.5, 0.05, 5.0)
    grad_clip = clip({self.clip_activation}({self.clip_w:.6f}*{self.clip_feature} + {self.clip_b:.6f}), 0.5, 10.0)
    noise_scale = clip({self.noise_activation}({self.noise_w:.6f}*{self.noise_feature} + {self.noise_b:.6f}) * 0.01, 0.0, 0.05)
    return {{"lr_scale": lr_scale, "grad_clip": grad_clip, "noise_scale": noise_scale}}
'''

    def mutate(self, rng: random.Random, scale: float = 0.08, allow_activation: bool = True, allow_feature: bool = True) -> "PolicyGenome":
        g = PolicyGenome(**asdict(self))
        choice = rng.choice(["coeff", "bias", "activation", "feature"])
        if choice == "coeff":
            field = rng.choice(["lr_w", "clip_w", "noise_w"])
            setattr(g, field, getattr(g, field) + rng.gauss(0.0, scale))
        elif choice == "bias":
            field = rng.choice(["lr_b", "clip_b", "noise_b"])
            setattr(g, field, getattr(g, field) + rng.gauss(0.0, scale))
        elif choice == "activation" and allow_activation:
            field = rng.choice(["lr_activation", "clip_activation", "noise_activation"])
            options = [a for a in ACTIVATIONS if a != getattr(g, field)]
            setattr(g, field, rng.choice(options))
        elif choice == "feature" and allow_feature:
            field = rng.choice(["lr_feature", "clip_feature", "noise_feature"])
            options = [f for f in FEATURES if f != getattr(g, field)]
            setattr(g, field, rng.choice(options))
        else:
            return g.mutate(rng, scale, allow_activation, allow_feature)
        return g


def validate_policy_source(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Forbidden AST node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES and node.id != "features" and node.id != "policy":
            raise ValueError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in ALLOWED_NAMES:
                raise ValueError(f"Forbidden call: {node.func.id}")
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct whitelisted function calls are allowed")


def compile_policy(source: str) -> Callable[[Dict[str, float]], Dict[str, float]]:
    validate_policy_source(source)
    env = {
        "sigmoid": sigmoid,
        "tanh": tanh,
        "relu": relu,
        "softplus": softplus,
        "clip": clip,
        "min": min,
        "max": max,
        "abs": abs,
        "float": float,
        "__builtins__": {},
    }
    local_env: Dict[str, object] = {}
    exec(compile(source, "<policy>", "exec"), env, local_env)
    fn = local_env.get("policy")
    if not callable(fn):
        raise ValueError("Compiled policy does not expose callable 'policy'")
    return fn
