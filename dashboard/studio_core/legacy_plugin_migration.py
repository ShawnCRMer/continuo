"""Expose the shared legacy authority resolver to independently loaded domains."""
import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location(
    'film_suite_legacy_migration_domain',
    Path(__file__).resolve().parents[1] / 'generation' / 'legacy_plugin_migration.py',
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
state_root = _module.state_root
TOMBSTONE_PATTERN = _module.TOMBSTONE_PATTERN
MigrationConflict = _module.MigrationConflict
