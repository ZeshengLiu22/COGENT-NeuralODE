"""Model factory and exports."""

from .node2_model import NODE2Model


def build_model(config: dict, num_static: int, num_force: int, num_state: int) -> NODE2Model:
    """Instantiate the latent-space controlled graph NODE."""

    return NODE2Model(num_static, num_force, num_state, config)


__all__ = ["NODE2Model", "build_model"]
