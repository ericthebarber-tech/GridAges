# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add project root to sys.path so autodoc can find gridages
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
project = "GridAges"
copyright = "2026, Hepeng Li"
author = "Hepeng Li"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"

# Napoleon settings for Google/NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_title = "GridAges Documentation"

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#1976d2",
        "color-brand-content": "#1565c0",
    },
    "dark_css_variables": {
        "color-brand-primary": "#64b5f6",
        "color-brand-content": "#90caf9",
    },
}

html_css_files = [
    "grid_builder.css",
]

html_js_files = [
    "grid_builder.js",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "tasklist",
    "html_image",
    "attrs_inline",
    "attrs_block",
]

master_doc = "index"
