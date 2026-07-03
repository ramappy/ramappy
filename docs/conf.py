# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import sys
from pathlib import Path

import sphinx.ext.viewcode

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# Project information
project = "ramappy"
copyright = "2026, Andrea Masella, Elia Broggio"
author = "Andrea Masella, Elia Broggio"

# General configuration
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_design",
    "sphinxarg.ext",
    "registry_index",
]

# AutoAPI: auto-generates API reference from source
autoapi_dirs = ["../ramappy"]
autoapi_type = "python"
autoapi_root = "reference/api"
autoapi_add_toctree_entry = False  # we add it manually in reference/index.md
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "class"
autoapi_member_order = "groupwise"
autoapi_keep_files = True
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
    # Exclude model internals and BaseModel noise from pydantic classes
    "exclude-members": "model_config,model_fields,model_computed_fields,model_post_init",
}

autodoc_class_signature = "mixed"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True

# ---------------------------------------------------------------------------
# autodoc-pydantic: render Pydantic models with full field metadata
# ---------------------------------------------------------------------------
autodoc_pydantic_model_show_json = False
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_config_members = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = True
# Show fields inherited from StepParams mixin bases (ParamsAcceptMask etc.)
autodoc_pydantic_model_members = True
autodoc_pydantic_model_inherited_members = True
autodoc_pydantic_model_undoc_members = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_doc_policy = "both"
autodoc_pydantic_field_show_constraints = True
autodoc_pydantic_field_show_default = True
autodoc_pydantic_field_show_alias = True
autodoc_pydantic_field_show_required = True
autodoc_pydantic_field_show_optional = True

# Napoleon: NumPy docstring support (no Google)
napoleon_numpy_docstring = True
napoleon_google_docstring = False
# napoleon_use_param = False
# napoleon_use_rtype = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
]
myst_heading_anchors = 3

# Intersphinx - cross-references to external libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# File extensions
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_title = "ramappy"
html_short_title = "ramappy"

# ---------------------------------------------------------------------------
# pydata_sphinx_theme configuration
# ---------------------------------------------------------------------------
html_theme_options = {
    # No "text" key here on purpose: the SVG itself already renders the
    # word "ramappy", so adding text would just duplicate it in the navbar.
    "logo": {
        "image_light": "assets/ramappy_logo.svg",
        "image_dark": "assets/ramappy_logo.svg",
    },
    "navigation_with_keys": True,
    "show_prev_next": False,
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button-field"],
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "use_edit_page_button": False,
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
    "back_to_top_button": True,
    "pygments_light_style": "friendly",
    "pygments_dark_style": "github-dark",
    "icon_links": [{"name": "GitHub", "url": "https://github.com/ramappy/ramappy", "icon": "fab fa-github"}],
}

_no_left_sidebar = [
    "getting-started/installation",
    "getting-started/quickstart",
    "user-guide/building-pipelines",
    "user-guide/extending-ramappy",
    "reference/steps",
    "reference/io-formats",
    "examples/index",
    "cli/batch",
    "license",
]
html_sidebars = {
    "index": [],
    **{page: [] for page in _no_left_sidebar},
    "**": ["sidebar-nav-bs"],
}

html_static_path = ["assets"]
html_css_files = ["custom.css"]
html_favicon = "assets/ramapp_icon.svg"


# Backstop for Pydantic's __init__/model_validate docstring boilerplate leaking
# into classes via inherited members (autoapi_python_class_content = "class"
# handles the common case; this catches the rest, e.g., mixins in a multiple-
# inheritance chain that still expose it).
_PYDANTIC_BOILERPLATE = [
    re.compile(
        r"Raises \[ValidationError\]\[pydantic_core\.ValidationError\] if the input data "
        r"cannot be validated to form a valid model\.?"
    ),
    re.compile(r"self is explicitly positional-only to allow self as a field name\.?"),
    re.compile(r"Create a new model by parsing and validating input data from keyword arguments\.?"),
]


def remove_mkdocs_admonitions(app, what, name, obj, options, lines):
    """Remove MkDocs-style admonitions (like !!!) and their indented content."""
    # Clear inherited Pydantic BaseModel docstrings to prevent duplicate/malformed output
    doc_str = "".join(lines)
    if "A base class for creating Pydantic models." in doc_str or "__pydantic_decorators__:" in doc_str:
        lines.clear()
        return

    text = "\n".join(lines)
    stripped = text
    for pattern in _PYDANTIC_BOILERPLATE:
        stripped = pattern.sub("", stripped)
    if stripped != text:
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        lines[:] = stripped.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("!!!"):
            lines.pop(i)
            # Remove all subsequent indented lines and trailing empty lines
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip() == "" or next_line.startswith(" ") or next_line.startswith("\t"):
                    lines.pop(i)
                else:
                    break
        else:
            i += 1


def skip_member(app, what, name, obj, skip, options):
    """Skip internal Pydantic models attributes/methods to keep API reference clean."""
    short_name = name.split(".")[-1]
    if short_name.startswith("__pydantic_") or short_name in {
        "__class_vars__",
        "__private_attributes__",
        "__signature__",
        "model_config",
        "model_fields",
        "model_computed_fields",
        "model_post_init",
    }:
        return True
    # Hide the IOParams base class itself from the API reference. Its fields
    # still show up on subclasses because autodoc_pydantic_model_inherited_members
    # is on — this only removes IOParams's own standalone page/entry.
    if what == "class" and short_name in {"IOParams", "BaseModel"}:
        return True
    return skip


def setup(app):
    app.connect("autodoc-process-docstring", remove_mkdocs_admonitions)
    app.connect("autoapi-skip-member", skip_member)


# Monkeypatch sphinx.ext.viewcode to fix IndexError on Python 3.14
def patched_collect_pages(app):
    import operator
    import posixpath

    import sphinx.ext.viewcode
    from sphinx.locale import _, __
    from sphinx.util.display import status_iterator

    env = app.env
    if not hasattr(env, "_viewcode_modules"):
        return
    if not sphinx.ext.viewcode.is_supported_builder(env._builder_cls, env.config.viewcode_enable_epub):
        return
    highlighter = app.builder.highlighter
    urito = app.builder.get_relative_uri

    modnames = set(env._viewcode_modules)

    for modname, entry in status_iterator(
        sorted(env._viewcode_modules.items()),
        __("highlighting module code... "),
        "blue",
        len(env._viewcode_modules),
        app.config.verbosity,
        operator.itemgetter(0),
    ):
        if not entry:
            continue
        if not sphinx.ext.viewcode.should_generate_module_page(app, modname):
            continue

        code, tags, used, refname = entry
        pagename = posixpath.join(sphinx.ext.viewcode.OUTPUT_DIRNAME, modname.replace(".", "/"))
        lexer = env.config.highlight_language if env.config.highlight_language in {"default", "none"} else "python"
        linenos = "inline" * env.config.viewcode_line_numbers
        highlighted = highlighter.highlight_block(code, lexer, linenos=linenos)
        lines = highlighted.splitlines()
        if not lines:
            continue
        try:
            before, after = lines[0].split("<pre>")
            lines[0:1] = [before + "<pre>", after]
        except ValueError:
            pass
        max_index = len(lines) - 1
        link_text = _("[docs]")
        for name, docname in used.items():
            if name not in tags:
                continue
            _type, start, end = tags[name]
            # Bounds check to avoid IndexError: list index out of range (Sphinx/Python 3.14 bug)
            if start < 0 or start > max_index:
                continue
            backlink = urito(pagename, docname) + "#" + refname + "." + name
            lines[start] = (
                f'<div class="viewcode-block" id="{name}">\n'
                f'<a class="viewcode-back" href="{backlink}">{link_text}</a>\n' + lines[start]
            )
            lines[min(end, max_index)] += "</div>\n"

        parents = []
        parent = modname
        while "." in parent:
            parent = parent.rsplit(".", 1)[0]
            if parent in modnames:
                parents.append({
                    "link": urito(
                        pagename,
                        posixpath.join(sphinx.ext.viewcode.OUTPUT_DIRNAME, parent.replace(".", "/")),
                    ),
                    "title": parent,
                })
        parents.append({
            "link": urito(pagename, posixpath.join(sphinx.ext.viewcode.OUTPUT_DIRNAME, "index")),
            "title": _("Module code"),
        })
        parents.reverse()
        context = {
            "parents": parents,
            "title": modname,
            "body": (_("<h1>Source code for %s</h1>") % modname + "\n".join(lines)),
        }
        yield pagename, context, "page.html"

    if not modnames:
        return

    html = ["\n"]
    stack = [""]
    for modname in sorted(modnames):
        if modname.startswith(stack[-1]):
            stack.append(modname + ".")
            html.append("<ul>")
        else:
            stack.pop()
            while not modname.startswith(stack[-1]):
                stack.pop()
                html.append("</ul>")
            stack.append(modname + ".")
        relative_uri = urito(
            posixpath.join(sphinx.ext.viewcode.OUTPUT_DIRNAME, "index"),
            posixpath.join(sphinx.ext.viewcode.OUTPUT_DIRNAME, modname.replace(".", "/")),
        )
        html.append(f'<li><a href="{relative_uri}">{modname}</a></li>')
    while len(stack) > 1:
        stack.pop()
        html.append("</ul>")
    context = {
        "parents": [],
        "title": _("Module code"),
        "body": (_("<h1>All modules for which code is available</h1>") + "".join(html)),
    }
    yield posixpath.join(sphinx.ext.viewcode.OUTPUT_DIRNAME, "index"), context, "page.html"


sphinx.ext.viewcode.collect_pages = patched_collect_pages
