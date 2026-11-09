from __future__ import annotations


_MUSIC_BRIDGE_PATH = Path(__file__).with_name("music_bridge.py")
_MUSIC_BRIDGE_SPEC = importlib.util.spec_from_file_location(
    "film_suite_music_bridge",
    _MUSIC_BRIDGE_PATH,
)
if _MUSIC_BRIDGE_SPEC is None or _MUSIC_BRIDGE_SPEC.loader is None:
    raise RuntimeError("The Music renderer bridge is unavailable.")
music_bridge = importlib.util.module_from_spec(_MUSIC_BRIDGE_SPEC)
_MUSIC_BRIDGE_SPEC.loader.exec_module(music_bridge)

_IMAGE_BRIDGE_PATH = Path(__file__).with_name("image_bridge.py")
_IMAGE_BRIDGE_SPEC = importlib.util.spec_from_file_location(
    "film_suite_image_bridge",
    _IMAGE_BRIDGE_PATH,
)
if _IMAGE_BRIDGE_SPEC is None or _IMAGE_BRIDGE_SPEC.loader is None:
    raise RuntimeError("The Image renderer bridge is unavailable.")
image_bridge = importlib.util.module_from_spec(_IMAGE_BRIDGE_SPEC)
_IMAGE_BRIDGE_SPEC.loader.exec_module(image_bridge)

_private_operations_module = _load_private_operations_module()
PrivateOperations = _private_operations_module.PrivateOperations
PrivateOperationError = _private_operations_module.PrivateOperationError
