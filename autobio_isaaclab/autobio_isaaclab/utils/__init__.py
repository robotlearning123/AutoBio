"""Utility functions for AutoBio Isaac Lab environments."""

import importlib
import pkgutil


def import_packages(package_name: str, blacklist: list[str] | None = None):
    """Recursively import all subpackages to trigger gym registration."""
    if blacklist is None:
        blacklist = []
    package = importlib.import_module(package_name)
    for importer, modname, ispkg in pkgutil.walk_packages(
        path=package.__path__, prefix=package.__name__ + ".", onerror=lambda x: None
    ):
        if any(bl in modname for bl in blacklist):
            continue
        try:
            importlib.import_module(modname)
        except Exception:
            pass
