# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = 'Astrolab Documentation'
copyright = '2023, Philip Cherian'
author = 'Philip Cherian'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'nbsphinx'
]

# NOTE: 'sphinx_tabs.tabs' conflicts with 'nbsphinx', causing notebook cells to not being rendered correctly.

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
pygments_style = 'solarized-light'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# Logo
html_logo = "_static/logos/astrolab-logo.svg"

# Favicon
html_favicon = '_static/logos/astrolab-favicon.ico'

# If true, “Created using Sphinx” is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, “(C) Copyright …” is shown in the HTML footer. Default is True.
html_show_copyright = False

# Show last updated as 'dd monthname yyyy'
html_last_updated_fmt = '%B %d, %Y'

# Theme Specific Options
# https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
html_theme_options = {
    'collapse_navigation': False,
    'logo_only': True,
    'navigation_depth': 5,
    'style_external_links': True,
    'style_nav_header_background': '#782121',
}

html_css_files = [
    'custom.css',
]

# MathJax configuration (uses LaTeX physics package)

mathjax3_config = {
      'tex': {
        'packages': ['base', 'ams', 'physics'],
        'inlineMath': [ ['$','$'], ["\\(","\\)"] ]
      },
      'loader': {
        'load': ['ui/menu', '[tex]/ams', '[tex]/physics']
      }
}
todo_include_todos = True

# Stops someone from closing the tab by clicking on it while open
# Basically, you can only cycle between tabs and never remove them from the display
sphinx_tabs_disable_tab_closing = True

# Sort functions by the order they appear in the source (for autodoc)
autodoc_member_order = 'bysource'

# Ask nbsphinx to never execute notebooks before rendering them on the website (since data will not be stored on the git repo).
nbsphinx_execute = 'never'
