"""The project's screenplay document ``studio.script/1`` (pure: no I/O but ``store``).

``schema``  shape, bounds and the derived views (numbers, aside, characters, places).
``fountain`` the Fountain reader/writer: scene text is Fountain; the full script with
            one ``[[s-xxxxxxxx]]`` note per scene for the agent; plain export and import.
``fdx``     Final Draft (FDX) import and export.
``ops``     one function per hand action, with per-scene conflict rules.
``store``   ``<project>/script/script.json`` + kept revisions + ``script.fountain``.
"""
