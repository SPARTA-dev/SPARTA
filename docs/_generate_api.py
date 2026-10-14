"""Generate docs/api.md: the API reference of sparta, unicor and todcor.

Reads the public surface (each module's ``__all__``, or the public functions
and classes it defines) with ``inspect`` and renders every signature and the
numpydoc summary, Parameters, Returns and Raises sections as Markdown. No
dependency beyond the packages themselves.

    python docs/_generate_api.py           # print to stdout
    python docs/_generate_api.py --write   # rewrite docs/api.md

``tests/test_docs_api.py`` fails when docs/api.md differs from this output.
"""
import dataclasses
import importlib
import inspect
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Document this checkout, whatever happens to be installed.
for _path in ("packages/todcor/src", "packages/unicor/src", "src"):
    if str(ROOT / _path) not in sys.path:
        sys.path.insert(0, str(ROOT / _path))

OUTPUT = ROOT / "docs" / "api.md"

# (heading, modules) in reading order; each module documents what it defines.
LAYOUT = (
    ("sparta", ("sparta.spectrum", "sparta.preprocess", "sparta.template", "sparta.ccf1d",
                "sparta.ccf2d", "sparta.tiravel", "sparta.observations", "sparta.timeseries",
                "sparta.periodogram", "sparta.pdc", "sparta.tiravel_ops",
                "sparta.instruments", "sparta.plotting")),
    ("unicor", ("unicor.ccf", "unicor.dsp")),
    ("todcor", ("todcor.estimator", "todcor.simulate")),
)
# Packages whose ``__all__`` re-exports the functions of the modules above.
PACKAGES = ("unicor", "todcor")
SECTIONS = ("Parameters", "Returns", "Raises")
WIDTH = 88

_ROLE = re.compile(r":(?:func|meth|class|mod|attr|data|obj|exc):`~?([^`]+)`")
_CODE = re.compile(r"``(.+?)``|`([^`]+)`")
# reST emphasis: asterisks hugging a word, not arithmetic such as 2*pi*zeta.
_EMPHASIS = re.compile(r"(?<![\w*])(\*{1,2})(?=[A-Za-z])[^*]*?[A-Za-z.)](\1)(?![\w*])")


# =============================================================================
# The public surface
# =============================================================================
def public_members(module):
    '''
    The public functions and classes a module defines, in source order.

    Parameters
    ----------
    module : module

    Returns
    -------
    list of (str, object)
        Names and objects: the functions and classes in the module's
        ``__all__`` (re-exports included, in ``__all__`` order) if it has
        one, else every public function and class it defines, in source
        order.
    '''
    names = getattr(module, "__all__", None)
    own = names is None
    if own:
        names = [n for n in vars(module) if not n.startswith("_")]
    members = []
    for name in names:
        obj = getattr(module, name)
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and (
                not own or obj.__module__ == module.__name__):
            members.append((name, obj))
    if own:
        members.sort(key=lambda item: _source_line(item[1]))
    return members


def data_members(module):
    '''
    The public non-callable names of a module (constants, registries).

    Parameters
    ----------
    module : module

    Returns
    -------
    list of str
        The names in ``__all__`` that are neither functions, classes nor
        modules; for a module without ``__all__``, the upper-case names it
        assigns at top level (not those it imports).
    '''
    names = getattr(module, "__all__", None)
    if names is None:
        source = inspect.getsource(module)
        names = [n for n in vars(module) if n.isupper() and not n.startswith("_")
                 and re.search(rf"^{n}\s*[:=]", source, re.MULTILINE)]
    return [n for n in names
            if not (inspect.isfunction(getattr(module, n)) or inspect.isclass(getattr(module, n))
                    or inspect.ismodule(getattr(module, n)))]


def class_members(cls):
    '''
    The public methods and properties a class defines, in source order.

    Parameters
    ----------
    cls : type

    Returns
    -------
    list of (str, object)
        Names and the raw class attributes (functions, classmethods,
        staticmethods, properties).
    '''
    out = []
    for name, attr in vars(cls).items():
        if name.startswith("_"):
            continue
        if isinstance(attr, (classmethod, staticmethod, property)) or inspect.isfunction(attr):
            out.append((name, attr))
    return out


def _source_line(obj):
    try:
        return inspect.getsourcelines(obj)[1]
    except (OSError, TypeError):
        return 0


# =============================================================================
# Signatures
# =============================================================================
def signature(name, obj, drop_first=False):
    '''
    A call signature without annotations, wrapped one parameter per line when
    long.

    Parameters
    ----------
    name : str
        The name to show (e.g. ``Spectrum.interpolate``).
    obj : callable
    drop_first : bool
        Drop the first parameter (``self`` of a plain method).

    Returns
    -------
    str
    '''
    sig = inspect.signature(obj)
    params = [p.replace(annotation=inspect.Parameter.empty) for p in sig.parameters.values()]
    if drop_first:
        params = params[1:]
    sig = sig.replace(parameters=params, return_annotation=inspect.Signature.empty)
    text = f"{name}{sig}"
    if len(text) <= WIDTH:
        return text
    parts = [str(p) for p in params]
    # Re-insert the markers a one-line str(sig) would carry.
    kinds = [p.kind for p in params]
    rendered = []
    for i, (part, kind) in enumerate(zip(parts, kinds)):
        if kind is inspect.Parameter.KEYWORD_ONLY and (
                i == 0 or kinds[i - 1] not in (inspect.Parameter.KEYWORD_ONLY,
                                               inspect.Parameter.VAR_POSITIONAL)):
            rendered.append("*")
        rendered.append(part)
        if kind is inspect.Parameter.POSITIONAL_ONLY and (
                i + 1 == len(kinds) or kinds[i + 1] is not inspect.Parameter.POSITIONAL_ONLY):
            rendered.append("/")
    return f"{name}(\n" + "".join(f"    {r},\n" for r in rendered) + ")"


# =============================================================================
# numpydoc parsing
# =============================================================================
def split_doc(doc):
    '''
    Split a cleaned docstring into its summary, warnings and sections.

    Parameters
    ----------
    doc : str or None
        ``inspect.getdoc`` output.

    Returns
    -------
    summary : str
        The first paragraph, on one line.
    warnings : list of str
        The text of each ``.. warning::`` directive.
    sections : dict of str to list of str
        The lines of each numpydoc section (header and underline removed).
    '''
    if not doc:
        return "", [], {}
    lines = doc.splitlines()
    first = []
    for line in lines:
        if not line.strip():
            break
        first.append(line.strip())
    summary = " ".join(first)

    warnings, sections, current = [], {}, None
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and line.strip() and re.fullmatch(r"-{3,}", lines[i + 1].strip()) \
                and not line.startswith(" "):
            current = line.strip()
            sections[current] = []
            i += 2
            continue
        if line.strip().startswith(".. warning::"):
            text = [line.strip()[len(".. warning::"):].strip()]
            i += 1
            while i < len(lines) and (lines[i].startswith(" ") or not lines[i].strip()):
                if not lines[i].strip() and (i + 1 == len(lines) or not lines[i + 1].startswith(" ")):
                    break
                text.append(lines[i].strip())
                i += 1
            warnings.append(" ".join(t for t in text if t))
            continue
        if current is not None:
            sections[current].append(line)
        i += 1
    return summary, warnings, sections


def entries(lines):
    '''
    The entries of a Parameters / Returns / Raises section.

    Parameters
    ----------
    lines : list of str
        The section body.

    Returns
    -------
    list of (str, list of str)
        Each entry's header line and its description lines, dedented. The
        section ends at the first unindented line that is prose rather than
        an entry header (numpydoc allows free text after a section).
    '''
    out = []
    for line in lines:
        if line.strip() and not line.startswith(" "):
            if not _is_header(line.strip()):
                break
            out.append((line.strip(), []))
        elif out:
            out[-1][1].append(line)
    return [(head, _dedent(body)) for head, body in out]


def _is_header(line):
    '''An entry header: ``name : type``, a list of names, or a short type.'''
    return (" : " in line or re.fullmatch(r"[\w*]+(, [\w*]+)*", line) is not None
            or (len(line.split()) <= 6 and not line.endswith((".", ":"))))


def _dedent(lines):
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    cut = min(indents) if indents else 0
    return [l[cut:] if l.strip() else "" for l in lines]


# =============================================================================
# Markdown
# =============================================================================
def inline(text):
    '''
    Convert reST inline markup to Markdown.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Roles and double-backtick literals as code spans; asterisks outside
        code spans escaped, so arithmetic is not read as emphasis.
    '''
    text = _ROLE.sub(lambda m: f"`{m.group(1)}`", text)
    out, pos = [], 0
    for m in _CODE.finditer(text):
        out.append(_prose(text[pos:m.start()]))
        out.append(f"`{m.group(1) or m.group(2)}`")
        pos = m.end()
    out.append(_prose(text[pos:]))
    return "".join(out)


def _prose(text):
    '''Keep reST ``*emphasis*`` and ``**strong**``; escape every other asterisk.'''
    out, pos = [], 0
    for m in _EMPHASIS.finditer(text):
        out.append(text[pos:m.start()].replace("*", r"\*"))
        out.append(m.group(0))
        pos = m.end()
    out.append(text[pos:].replace("*", r"\*"))
    return "".join(out).replace("<", r"\<")


def render_block(lines):
    '''
    Render a dedented description as Markdown lines.

    Lines at the base indent form paragraphs. A line directly followed by
    deeper-indented lines is a term of a definition list (a bullet); a
    deeper-indented block after a blank line is nested content, shown as code
    when it holds no terms (a formula).

    Parameters
    ----------
    lines : list of str

    Returns
    -------
    list of str
        Markdown lines, unindented, blocks separated by blank lines.
    '''
    blocks, para, i = [], [], 0

    def flush():
        if para:
            blocks.append(("para", re.sub(r" {2,}", " ", " ".join(p.strip() for p in para))))
            para.clear()

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
            i += 1
            continue
        if line.startswith(" "):
            # Nested content after a blank line (a term's body is consumed below).
            j = i
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith(" ")):
                j += 1
            body = _dedent(lines[i:j])
            sub = render_block(body)
            if any(s.startswith("- ") for s in sub):
                blocks.append(("nested", sub))
            else:
                blocks.append(("literal", body))
            i = j
            continue
        if i + 1 < len(lines) and lines[i + 1].startswith(" ") and lines[i + 1].strip():
            flush()
            j = i + 1
            while j < len(lines) and (lines[j].startswith(" ") or (
                    not lines[j].strip() and j + 1 < len(lines) and lines[j + 1].startswith(" "))):
                j += 1
            blocks.append(("term", line.strip(), render_block(_dedent(lines[i + 1:j]))))
            i = j
            continue
        para.append(line)
        i += 1
    flush()

    out = []
    for block in blocks:
        kind = block[0]
        if kind == "para":
            out += [inline(block[1]), ""]
        elif kind == "nested":
            out += block[1] + [""]
        elif kind == "literal":
            out += ["```text"] + [l for l in block[1]] + ["```", ""]
        else:
            out += _bullet(inline(block[1]), block[2])
    return _tighten(out)


def _bullet(head, body):
    '''A list item: ``head`` and the first paragraph of ``body`` on one line,
    the rest of ``body`` indented beneath it.'''
    if body and body[0] and not body[0].startswith(("- ", "```")):
        rest = body[1:]
        while rest and not rest[0]:
            rest = rest[1:]
        first = f"- {head}: {body[0]}" if head else f"- {body[0]}"
        lines = [first] + (([""] + rest) if rest else [])
    else:
        lines = [f"- {head}"] + ([""] + body if body else [])
    return [lines[0]] + [("  " + l) if l else "" for l in lines[1:]] + [""]


def render_sections(sections):
    '''
    Render the Parameters, Returns and Raises sections.

    Parameters
    ----------
    sections : dict of str to list of str
        From :func:`split_doc`.

    Returns
    -------
    list of str
        Markdown lines.
    '''
    out = []
    for name in ("Attributes",) + SECTIONS:
        if name not in sections:
            continue
        items = entries(sections[name])
        if not items:
            continue
        out += [f"**{name}**", ""]
        for head, body in items:
            if " : " in head:
                names, kind = head.split(" : ", 1)
                label = f"`{names}` (*{inline(kind)}*)"
            elif name in ("Returns", "Raises"):
                label = f"*{inline(head)}*"
            else:
                label = f"`{head}`"
            out += _bullet(label, render_block(body))
        out = _tighten(out)
        out.append("")
    return out


def _tighten(lines):
    '''Drop the blank lines between consecutive bullets (one list) and the
    trailing blank lines.'''
    out = []
    for k, line in enumerate(lines):
        if not line and out and k + 1 < len(lines) and lines[k + 1].startswith("- ") \
                and (out[-1].startswith("- ") or out[-1].startswith("  ")):
            continue
        out.append(line)
    while out and not out[-1]:
        out.pop()
    return out


def render_doc(doc, title, sig=None, level=3, extra=None):
    '''
    One documented object: heading, signature, summary, warnings, sections.

    Parameters
    ----------
    doc : str or None
        Its cleaned docstring.
    title : str
        The heading text.
    sig : str, optional
        The call signature, shown as a Python code block.
    level : int
        Heading level.
    extra : dict of str to list of str, optional
        Sections taken from another docstring (a class's ``__init__``),
        rendered after the object's own.

    Returns
    -------
    list of str
        Markdown lines.
    '''
    summary, warnings, sections = split_doc(doc)
    out = [f"{'#' * level} {title}", ""]
    if sig:
        out += ["```python", sig, "```", ""]
    if summary:
        out += [inline(summary), ""]
    for text in warnings:
        out += [f"> **Warning.** {inline(text)}", ""]
    out += render_sections(sections)
    if extra:
        out += render_sections(extra)
    while out and not out[-1]:
        out.pop()
    return out + [""]


def render_class(cls, display, level=4):
    '''
    A class: its constructor and every public method and property.

    Parameters
    ----------
    cls : type
    display : str
        The name to show, e.g. ``sparta.Spectrum``.
    level : int
        Heading level of the class; its members are one level below.

    Returns
    -------
    list of str
        Markdown lines.
    '''
    init = vars(cls).get("__init__")
    init_doc = inspect.getdoc(init) if init is not None and not dataclasses.is_dataclass(cls) \
        else None
    _, _, init_sections = split_doc(init_doc)
    out = render_doc(inspect.getdoc(cls), f"`{display}`",
                     signature(cls.__name__, cls), level=level, extra=init_sections)
    for name, attr in class_members(cls):
        qual = f"{cls.__name__}.{name}"
        if isinstance(attr, property):
            out += render_doc(inspect.getdoc(attr.fget), f"`{qual}` (property)", level=level + 1)
        elif isinstance(attr, staticmethod):
            out += render_doc(inspect.getdoc(attr), f"`{qual}`",
                              signature(qual, attr.__func__), level=level + 1)
        elif isinstance(attr, classmethod):
            out += render_doc(inspect.getdoc(attr), f"`{qual}` (classmethod)",
                              signature(qual, getattr(cls, name)), level=level + 1)
        else:
            out += render_doc(inspect.getdoc(attr), f"`{qual}`",
                              signature(qual, attr, drop_first=True), level=level + 1)
    return out


def slug(heading):
    '''
    The anchor GitHub gives a heading.

    Parameters
    ----------
    heading : str

    Returns
    -------
    str
        Lower case, punctuation other than '-' and '_' removed, spaces as '-'.
    '''
    return re.sub(r"[^a-z0-9 _-]", "", heading.lower()).replace(" ", "-")


def exported_as(obj, module):
    '''
    The shortest public import path of a module-level object.

    Parameters
    ----------
    obj : function or class
    module : module
        The module whose section documents it.

    Returns
    -------
    str
        ``package.name`` when the object is in the ``__all__`` of its
        top-level package (``sparta``, ``unicor`` or ``todcor``), else
        ``module.name``.
    '''
    top = module.__name__.split(".")[0]
    package = importlib.import_module(top)
    if getattr(package, obj.__name__, None) is obj and (
            obj.__name__ in getattr(package, "__all__", ())):
        return f"{top}.{obj.__name__}"
    return f"{module.__name__}.{obj.__name__}"


def render():
    '''
    The content of docs/api.md.

    Returns
    -------
    str
    '''
    body, toc = [], []
    for package, modules in LAYOUT:
        toc.append(f"- **{package}**")
        body += [f"## {package}", ""]
        pkg = importlib.import_module(package)
        pkg_summary = split_doc(inspect.getdoc(pkg))[0]
        if pkg_summary:
            body += [inline(pkg_summary), ""]
        if package in PACKAGES:
            body += [f"Everything below is importable from `{package}` itself "
                     f"(`{package}.__all__`).", ""]
        if package == "todcor":
            reexports = [n for n in pkg.__all__
                         if getattr(getattr(pkg, n), "__module__", "").startswith("unicor")
                         or n == "C_KMS"]
            body += ["`todcor` also re-exports these `unicor.dsp` names, documented under "
                     "[`unicor.dsp`](#unicordsp): "
                     + ", ".join(f"`{n}`" for n in reexports) + ".", ""]
        for name in modules:
            module = importlib.import_module(name)
            members = public_members(module)
            classes = [n for n, o in members if inspect.isclass(o)]
            toc.append(f"  - [`{name}`](#{slug(name)})"
                       + (" — " + ", ".join(f"`{c}`" for c in classes) if classes else ""))
            body += [f"### {name}", ""]
            summary = split_doc(inspect.getdoc(module))[0]
            if summary:
                body += [inline(summary), ""]
            constants = []
            for d in data_members(module):
                value = getattr(module, d)
                if isinstance(value, dict) and hasattr(module, "__all__"):
                    body += [f"`{name}.{d}`: a dict keyed by "
                             + ", ".join(f"`{k}`" for k in value) + ".", ""]
                else:
                    constants.append(f"`{d}` ({type(value).__name__})")
            if constants:
                body += [f"Constants in `{name}`: " + ", ".join(constants) + ".", ""]
            for member_name, obj in members:
                if inspect.isclass(obj):
                    body += render_class(obj, exported_as(obj, module))
                else:
                    shown = exported_as(obj, module)
                    body += render_doc(inspect.getdoc(obj), f"`{shown}`",
                                       signature(shown, obj), level=4)
    head = ["# API reference", "",
            "<!-- Generated by `python docs/_generate_api.py --write` from the docstrings; "
            "do not edit by hand. -->", "",
            "Signatures and the numpydoc summary, Parameters, Returns and Raises of every "
            "public class and function of `sparta` and of the two engines, `unicor` and "
            "`todcor`. Units are as stated in each entry: velocities in km/s, wavelengths "
            "in Å, times in days, frequencies in 1/day. The conventions behind them are in "
            "[conventions.md](conventions.md); the survey keys of `sparta.instruments` in "
            "[instruments.md](instruments.md).", "",
            "## Contents", ""] + toc + [""]
    text = "\n".join(head + body)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip("\n") + "\n"


if __name__ == "__main__":
    content = render()
    if "--write" in sys.argv[1:]:
        OUTPUT.write_text(content, encoding="utf-8")
    else:
        sys.stdout.write(content)
