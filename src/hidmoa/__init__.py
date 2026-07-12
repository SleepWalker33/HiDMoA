"""HiDMoA: class-incremental defect classification with a VAE task router."""


def run_hidmoa(*args, **kwargs):
    from .main import run_incremental_2

    return run_incremental_2(*args, **kwargs)

__all__ = ["run_hidmoa"]
__version__ = "0.1.0"
