import os
import uuid
import itertools
import mimetypes
import posixpath
from base64 import b64encode

import nbformat
from sphinx import roles, addnodes
from docutils import nodes, writers
from nbclient.exceptions import CellExecutionError
from docutils.parsers.rst import directives
from nbconvert.preprocessors import ExecutePreprocessor


def _cell_uuid():
    return uuid.uuid4().hex[:8]


def _finalize_markdown_cells(nb):
    markdown_cells = (c for c in nb.cells if c.cell_type == "markdown")
    for cell in markdown_cells:
        cell.source = "".join(cell.source).strip()

    return nb


class NBWriter(writers.Writer):
    """Jupyter notebook writer"""

    def __init__(self, app, docpath):
        super().__init__()
        self.app = app
        self.docpath = docpath

    def translate(self):
        visitor = NBTranslator(self.document, self.app, self.docpath)
        self.document.walkabout(visitor)
        nb = _finalize_markdown_cells(visitor.nb)

        if self.app.config.nbexport_execute:
            ep = ExecutePreprocessor(allow_errors=True)
            try:
                ep.preprocess(nb, {"metadata": {}})
            except CellExecutionError as e:
                self.app.warn(str(e))

        self.output = nbformat.writes(nb)


def _split_doctest(code):
    """Split a single doctest string into multiple code block strings"""

    def is_code(x):
        return x.startswith(">>>") or x.startswith("...")

    groups = itertools.groupby(code.splitlines(), is_code)
    raw_code_blocks = (lines for is_code, lines in groups if is_code)
    code_blocks = ["\n".join(line[4:].rstrip() for line in lines) for lines in raw_code_blocks]
    return code_blocks


def _image_url(imagepath):
    with open(imagepath, "rb") as f:
        image = f.read()
    image_str = b64encode(image).decode("utf-8")
    content_type = mimetypes.guess_type(imagepath)[0]
    image_url = f"data:{content_type};base64,{image_str}"
    return image_url


# noinspection PyPep8Naming,PyUnusedLocal,PyMethodMayBeStatic
class NBTranslator(nodes.NodeVisitor):
    def __init__(self, document, app, docpath):
        super().__init__(document)
        self.section_level = 0
        self.indent = 0
        self.paragraph_prefix = ""
        self.paragraph_suffix = ""
        self.paragraph_html = False

        self.app = app
        self.config = app.config
        self.docpath = docpath

        self.nb = nbformat.from_dict(
            {
                "cells": [],
                "metadata": {},
                "nbformat": nbformat.current_nbformat,
                "nbformat_minor": nbformat.current_nbformat_minor,
            }
        )
        if self.config.nbexport_pre_code:
            self.write_code(self.config.nbexport_pre_code)

    def write_markdown(self, text):
        if self.nb.cells[-1].cell_type != "markdown":
            self.nb.cells.append(
                nbformat.from_dict(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [],
                        "id": _cell_uuid(),
                    }
                )
            )

        self.nb.cells[-1].source.append(text.replace("\n", "\n" + " " * self.indent))

    def rstrip_markdown(self, chars=None):
        if self.nb.cells[-1].cell_type != "markdown":
            return
        self.nb.cells[-1].source[-1] = self.nb.cells[-1].source[-1].rstrip(chars)

    def add_image(self, id, image_url):
        attachments = self.nb.cells[-1].setdefault("attachments", {})
        encoding, image = image_url.split(",")
        encoding = encoding.split(":")[-1].split(";")[0]
        attachments[id] = {encoding: image}

    def add_codecell(self, code):
        self.nb.cells.append(
            nbformat.from_dict(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": code.strip(),
                    "outputs": [],
                    "id": _cell_uuid(),
                }
            )
        )

    def write_code(self, code):
        if ">>>" in code:
            for doctest in _split_doctest(code):
                self.add_codecell(doctest)
        else:
            self.add_codecell(code)

    def visit_section(self, node):
        self.section_level += 1

    def depart_section(self, node):
        self.section_level -= 1

    def visit_title(self, node):
        self.write_markdown("#" * self.section_level + " ")

    def depart_title(self, node):
        self.write_markdown("\n\n")

    def visit_Text(self, node):
        self.write_markdown(node.replace("\n", " "))

    def depart_Text(self, node):
        pass

    def visit_list_item(self, node):
        self.write_markdown("* ")
        self.indent += 2

    def depart_list_item(self, node):
        self.indent -= 2
        self.rstrip_markdown(" ")

    def visit_note(self, node):
        self.paragraph_prefix = '<div class="alert alert-block alert-info"><b>Note: </b>'
        self.paragraph_suffix = "</div>"
        self.paragraph_html = True

    def depart_note(self, node):
        self.paragraph_prefix = ""
        self.paragraph_suffix = ""
        self.paragraph_html = False

    def visit_warning(self, node):
        self.paragraph_prefix = '<div class="alert alert-block alert-danger"><b>Warning: </b>'
        self.paragraph_suffix = "</div>"
        self.paragraph_html = True

    def depart_warning(self, node):
        self.paragraph_prefix = ""
        self.paragraph_suffix = ""
        self.paragraph_html = False

    def visit_paragraph(self, node):
        if self.paragraph_prefix:
            self.write_markdown(self.paragraph_prefix)

    def depart_paragraph(self, node):
        if self.paragraph_suffix:
            self.write_markdown(self.paragraph_suffix)
        self.write_markdown("\n\n")

    def _grab_target(self, node):
        if "refuri" in node:
            url = node["refuri"]
            if node.get("internal"):
                return posixpath.join(self.config.nbexport_baseurl, self.docpath, url)
        else:
            return f"#{node['refid']}"

    def visit_reference(self, node):
        self.write_markdown(f'<a href="{self._grab_target(node)}">' if self.paragraph_html else "[")

    def depart_reference(self, node):
        self.write_markdown("</a>" if self.paragraph_html else f"]({self._grab_target(node)})")

    def visit_download_reference(self, node):
        if node.hasattr("filename"):
            self.write_markdown("[" + node.astext())
            raise nodes.SkipChildren

    def depart_download_reference(self, node):
        if node.hasattr("filename"):
            url = posixpath.join(self.config.nbexport_baseurl, "_downloads", node["filename"])
            self.write_markdown("]({})".format(url))

    def visit_literal(self, node):
        self.write_markdown("`")

    def depart_literal(self, node):
        self.write_markdown("`")

    def visit_literal_block(self, node):
        dont_execute = (
            "highlight_args" in node
            and "hl_lines" in node["highlight_args"]
            and 0 in node["highlight_args"]["hl_lines"]
        )
        if dont_execute:
            self.write_markdown("```{}\n{}\n```\n\n".format(node["language"], node.astext()))
        else:
            self.write_code(node.astext())
        raise nodes.SkipNode

    def visit_doctest_block(self, node):
        self.visit_literal_block(node)

    def visit_math(self, node):
        self.write_markdown("${}$".format(node.astext()))
        raise nodes.SkipNode

    def visit_math_block(self, node):
        self.write_markdown("$$\n{}\n$$\n\n".format(node.astext().strip()))
        raise nodes.SkipNode

    def visit_image(self, node):
        if "nbattach" in node and node["nbattach"]:
            filename = os.path.basename(node["uri"])
            self.write_markdown(f"![{filename}](attachment:{filename})\n\n")
            self.add_image(filename, _image_url(os.path.join(self.app.srcdir, node["uri"])))

    def unknown_visit(self, node):
        pass

    def unknown_departure(self, node):
        pass


def export_notebooks(app, document, docname):
    """Export the recently resolved document to a Jupyter notebook"""
    if not hasattr(app.env, "nbfiles"):
        return
    if docname not in app.env.nbfiles:
        return

    docpath = os.path.dirname(docname)
    ipynb_path = app.env.nbfiles[docname]
    with open(ipynb_path, "w", encoding="utf-8") as file:
        writer = NBWriter(app, docpath)
        writer.write(document, file)


def cleanup_notebooks(app, _):
    """Delete cache"""
    if hasattr(app.env, "nbfiles"):
        del app.env.nbfiles


def remove_notebooks_from_deps(app, _):
    """Remove notebook files from the sphinx dependency tracker

    The exported notebooks always have a later timestamp than the source document,
    so they would always appear as changed files. The notebooks are rebuilt anytime
    a document is updated anyway, so the dependency checking is not needed.
    """
    env = app.env
    if not hasattr(env, "nbfiles"):
        return

    for ipynb_path in env.nbfiles.values():
        for docname, deps in env.dependencies.items():
            docpath = os.path.dirname(docname)
            relpath = os.path.relpath(ipynb_path, os.path.join(env.srcdir, docpath))
            deppath = os.path.join(docpath, relpath)
            if deppath in deps:
                deps.remove(deppath)


def _make_empty_file(abspath):
    absdir = os.path.dirname(abspath)
    if not os.path.exists(absdir):
        os.makedirs(absdir)
    open(abspath, "w").close()


class NotebookExportRole(roles.XRefRole):
    """Mark a document for export and hold a reference to the exported notebook"""

    def process_link(self, env, refnode, has_explicit_title, title, target):
        if target == "self":
            target = "/" + env.docname
        target_docname, _ = env.relfn2path(target, env.docname)
        ipynb_name = target_docname + ".ipynb"

        if not has_explicit_title:
            title = ipynb_name

        build_dir = os.path.join(os.path.dirname(env.app.doctreedir), "nbexport")
        ipynb_abspath = os.path.join(build_dir, ipynb_name)
        _make_empty_file(ipynb_abspath)  # placeholder file so the reference can be resolved

        if not hasattr(env, "nbfiles"):
            env.nbfiles = {}
        env.nbfiles[target_docname] = ipynb_abspath  # doc to be converted -> abs path to notebook

        absdocpath = os.path.dirname(os.path.join(env.srcdir, env.docname))
        ipynb_relpath = os.path.relpath(ipynb_abspath, absdocpath).replace(os.path.sep, "/")
        return super().process_link(env, refnode, has_explicit_title, title, ipynb_relpath)

    def result_nodes(self, document, env, node, is_ref):
        node.children[0]["classes"].append("download")
        return super().result_nodes(document, env, node, is_ref)


class NBAttachImage(directives.images.Image):
    """Extend the `Image` directive class with the option `:nbattach:` to enable attaching an image
    to a notebook"""

    option_spec = directives.images.Image.option_spec.copy()
    option_spec["nbattach"] = directives.flag

    def run(self):
        self.options["nbattach"] = True if "nbattach" in self.options else False
        return super().run()


def setup(app):
    # Replace standard image Docutils directive with extended `NBAttachImage` enabling `:nbattach`
    app.add_directive("image", NBAttachImage, override=True)

    app.add_role("nbexport", NotebookExportRole(nodeclass=addnodes.download_reference))

    app.connect("doctree-resolved", export_notebooks)
    app.connect("build-finished", cleanup_notebooks)
    app.connect("doctree-read", remove_notebooks_from_deps)

    app.add_config_value("nbexport_pre_code", None, "html")
    app.add_config_value("nbexport_baseurl", "", "html")
    app.add_config_value("nbexport_execute", False, "html")

    return {"version": "0.2"}
