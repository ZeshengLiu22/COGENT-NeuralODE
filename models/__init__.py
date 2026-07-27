"""Model factory and exports."""

from .ncde1_model import NCDE1Model
from .node1_model import NODE1Model
from .node2_model import NODE2Model


def build_model(config: dict, num_static: int, num_force: int, num_state: int):
    """Instantiate one of the three required baseline models."""

    model_name = str(config["model"]["name"]).lower()
    if model_name == "node1":
        return NODE1Model(num_static, num_force, num_state, config)
    if model_name == "node2":
        return NODE2Model(num_static, num_force, num_state, config)
    if model_name == "ncde1":
        return NCDE1Model(num_static, num_force, num_state, config)
    raise ValueError(f"Unknown model name: {config['model']['name']!r}")


__all__ = ["NCDE1Model", "NODE1Model", "NODE2Model", "build_model"]
