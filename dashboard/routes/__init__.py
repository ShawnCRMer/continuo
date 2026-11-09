"""Area composition with the historical API namespace as compatibility boundary.

Hermes imports the dashboard API under an arbitrary top-level module name.
All extracted declarations execute in that module's namespace: mutable state,
legacy monkeypatches, Pydantic forward references, reload cleanup and __file__
relative dependency paths therefore retain their original meaning. This is not
an alternate runtime API; plugin_api remains the single public export surface.
"""
from importlib.machinery import SourceFileLoader
from pathlib import Path


def load_area(relative_path, namespace):
    """Load one trusted, repository-owned composition unit, never caller input."""
    source = Path(__file__).parent / relative_path
    code = SourceFileLoader(namespace['__name__'], str(source)).get_code(namespace['__name__'])
    if code is None:
        raise ImportError(f'Route composition unit is unavailable: {source}')
    exec(code, namespace)
