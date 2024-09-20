# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'scikit omikuji'
copyright = '2024, Carlo Nicolini'
author = 'Carlo Nicolini'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Configure autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# Configure Napoleon to parse Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Make sure the following settings are present:
autodoc_typehints = "description"
autodoc_member_order = 'bysource'

# Add the path to your source code
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
