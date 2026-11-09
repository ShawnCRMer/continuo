"""Film Suite timeline core: the ``studio.timeline/1`` document and its edit algebra.

Pure modules, no I/O except ``resolver`` (which only reads through injected
lookups). ``schema`` validates the document (TIMELINE-INPUT section 2.4),
``ops`` applies the verbs of section 3.2 by name. The renderer twin lives in
``desktop/film-suite/src/timeline/ops``; both run the shared fixture suite in
``tests/backend/fixtures/timeline``.
"""
