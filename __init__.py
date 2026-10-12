"""prestomorph.ui — the interactive screens. Components 3 and 5.

**Everything in this package may import napari and Qt. Nothing outside it may.** That is the
whole reason the package exists as a boundary rather than as a folder, and
``tests/test_ui_boundary.py`` asserts both halves.

PrestoCell's failure was not that it called ``napari.run()`` — every napari application does.
It was that it called it from *inside a pipeline function* (``main.py:1351``), so the pipeline
could not run without a human present, which is why it has no batch mode.

    The rule is the direction of the arrow: the UI calls the pipeline, and the pipeline never
    calls the UI. ``napari.run()`` must be unreachable from anything Phase A executes.

**This module imports nothing**, mirroring ``prestomorph/__init__.py`` — so ``import
prestomorph.ui`` pulls in no Qt and costs nothing. Import the screen you want.

**Flat, with no ``shared/`` subpackage.** ``prestomorph/`` itself is 21 flat modules including
groups with obvious cohesion, so five should be flat too. And the boundary a subpackage would
draw is the wrong one: ``ui/shared/`` implies a Component 3 / Component 5 split in the
filesystem, but that split is a *planning* boundary, not a runtime one — the two tools
deliberately share ``editing`` and ``viewer``.

``COMPONENT_03_PLAN.md`` §4a.7.
"""
