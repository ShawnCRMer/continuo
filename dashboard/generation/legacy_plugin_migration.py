"""Non-destructive F6 legacy state authority and config-token migration.

Private stores bind absolute paths, HMAC identities, file handles and SQLite WAL
state. Do NOT copy/move them: retaining the legacy authority as a read/write
fallback preserves pending operations, project references and all historical
bytes. New profiles use the model-neutral root. Two state trees fail closed
instead of picking/merging artist work. This module never creates/deletes data.
"""
from pathlib import Path
import stat

LEGACY_ID = 'flux3-console'
PLUGIN_ID = 'film-suite'
TOMBSTONE_PATTERN = r'^\.(?:film-suite|flux3-console)-tombstone-[0-9a-f]{32}\.bin$'

class MigrationConflict(RuntimeError):
    """Legacy and new state need explicit reconciliation, never overwrite."""

def _exists_regular(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise MigrationConflict('state migration refuses symlink/reparse roots')
    if not stat.S_ISDIR(info.st_mode):
        raise MigrationConflict('state migration requires directory roots')
    return True

# Re-creatable caches and installed engines only (no artist work, no identity):
# a current-id root that holds nothing else never outranks an existing legacy
# root. "engines" holds downloaded model runners (Beat This!), reinstallable.
CACHE_LEAVES = frozenset({'lora-library', 'style-thumbs', 'engines'})

def _cache_only(root):
    try:
        return all(entry.name in CACHE_LEAVES and entry.is_dir() and not entry.is_symlink() for entry in root.iterdir())
    except OSError:
        return False

def state_root(parent, *parts):
    """Resolve one private state surface without changing its byte authority.

    A leaf that exists in neither tree follows the existing legacy id root,
    so a new leaf never opens a second id root beside it (A7).
    """
    parent = Path(parent)
    legacy = parent / LEGACY_ID / Path(*parts)
    current = parent / PLUGIN_ID / Path(*parts)
    # Validate id ancestors too when resolving plugin-contained bridge leaves.
    legacy_root = _exists_regular(parent / LEGACY_ID)
    current_root = _exists_regular(parent / PLUGIN_ID)
    old_exists = _exists_regular(legacy)
    new_exists = _exists_regular(current)
    if old_exists and new_exists:
        if not parts and _cache_only(current):
            return legacy
        raise MigrationConflict('both legacy and Continuo state exist; reconcile without overwriting')
    if old_exists or new_exists:
        return legacy if old_exists else current
    if parts and legacy_root and (not current_root or _cache_only(parent / PLUGIN_ID)):
        return legacy
    return current

def config_bytes(data):
    """Change only exact plugin-list scalar tokens; keep all other bytes intact."""
    import yaml  # Config-only dependency; private path resolution remains stdlib.
    text = data.decode('utf-8')
    tree = yaml.compose(text)
    spans = set()
    def visit(node, path):
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                visit(value, path + (key.value,))
        elif isinstance(node, yaml.SequenceNode):
            if path in {('plugins', 'enabled'), ('plugins', 'disabled'),
                        ('dashboard', 'plugins', 'enabled'), ('dashboard', 'plugins', 'disabled')}:
                for item in node.value:
                    if isinstance(item, yaml.ScalarNode) and item.value == LEGACY_ID:
                        spans.add((item.start_mark.index, item.end_mark.index))
    if tree is not None:
        visit(tree, ())
    if spans and any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(text)):
        raise MigrationConflict('config aliases require explicit lossless reconciliation')
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + text[start:end].replace(LEGACY_ID, PLUGIN_ID) + text[end:]
    return text.encode('utf-8')
