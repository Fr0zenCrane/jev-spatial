"""Jev-Spatial: a shared classifier for structured spatial decisions."""

__version__ = "0.1.0"

__all__ = ["JevSpatial"]


def __getattr__(name):
    if name == "JevSpatial":
        from .inference import JevSpatial
        return JevSpatial
    raise AttributeError(name)
