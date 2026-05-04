"""Extraction subpackage overlay — extends installed package with BAML support.

This module replicates the __path__ extension trick from the top-level
neo4j_agent_memory/__init__.py so that BOTH overlay modules (baml_extractor,
baml_config, factory_ext) AND base package modules (base, factory,
llm_extractor, spacy_extractor, etc.) remain importable.

Without this, creating any .py file in this directory would shadow the
entire installed extraction package. See RFI-I1.
"""

import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# Recursion guard: the base package __init__.py may trigger this overlay
# again (especially on Python 3.14+). Bail early on re-entry.
# ---------------------------------------------------------------------------
_LOADING_SENTINEL = "_neo4j_extraction_overlay_loading"
if getattr(sys, _LOADING_SENTINEL, False):
    pass  # re-entrant — skip the overlay logic entirely
else:
    setattr(sys, _LOADING_SENTINEL, True)
    try:
        # -------------------------------------------------------------------
        # 1. Locate the installed extraction package in site-packages.
        # -------------------------------------------------------------------
        _overlay_dir = os.path.dirname(os.path.abspath(__file__))

        _installed_dir = None
        for _p in sys.path:
            _candidate = os.path.join(_p, "neo4j_agent_memory", "extraction")
            if (
                os.path.isdir(_candidate)
                and os.path.normpath(_candidate) != os.path.normpath(_overlay_dir)
                and os.path.isfile(os.path.join(_candidate, "__init__.py"))
            ):
                _installed_dir = _candidate
                break

        # -------------------------------------------------------------------
        # 2. Extend __path__: overlay first, installed second.
        # -------------------------------------------------------------------
        if _installed_dir:
            __path__ = [_overlay_dir, _installed_dir]
        else:
            __path__ = [_overlay_dir]

        # -------------------------------------------------------------------
        # 3. Execute the installed extraction __init__.py to preserve exports
        #    (EntityExtractor, ExtractedEntity, create_extractor, etc.).
        # -------------------------------------------------------------------
        if _installed_dir:
            _base_init = os.path.join(_installed_dir, "__init__.py")
            if os.path.isfile(_base_init):
                _spec = importlib.util.spec_from_file_location(
                    "neo4j_agent_memory.extraction._base_init",
                    _base_init,
                    submodule_search_locations=[_installed_dir],
                )
                if _spec and _spec.loader:
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    # Copy all public symbols
                    _base_all = getattr(_mod, "__all__", [])
                    for _name in _base_all:
                        if hasattr(_mod, _name):
                            globals()[_name] = getattr(_mod, _name)
                    # Preserve __all__
                    if _base_all:
                        __all__ = list(_base_all)
                    # Preserve __getattr__ for lazy imports
                    if hasattr(_mod, "__getattr__"):
                        _base_getattr = _mod.__getattr__

                        def __getattr__(name: str):
                            return _base_getattr(name)
    finally:
        setattr(sys, _LOADING_SENTINEL, False)
