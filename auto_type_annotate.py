from __future__ import annotations

import argparse
import ast
import collections
import json
import os.path
import re
import subprocess
import sys
import tempfile
from collections.abc import Generator
from collections.abc import Sequence
from typing import NamedTuple
from typing import Self
from typing import TypedDict

from tokenize_rt import reversed_enumerate
from tokenize_rt import src_to_tokens
from tokenize_rt import Token
from tokenize_rt import tokens_to_src

_PEP_585 = frozenset(('Dict', 'FrozenSet', 'List', 'Set', 'Tuple', 'Type'))
_PYUPGRADE_PREFIX = f'''\
import typing
from typing import Union, Optional, {", ".join(sorted(_PEP_585))}
x: '''


def _to_mod(fname: str, roots: tuple[str, ...]) -> str:
    relpaths = [os.path.relpath(fname, root) for root in roots]
    relpaths.sort(key=len)
    for relative in relpaths:
        assert not relative.startswith('..'), relative  # report a bug?
        return (
            relative.removesuffix('.py')
            .replace('/', '.')
            .replace('\\', '.')
            .removesuffix('.__init__')
        )
    else:
        raise AssertionError(f'{fname=} not found in {roots=}')


class Mod(NamedTuple):
    path: str
    modname: str

    @classmethod
    def from_path(cls, path: str, roots: tuple[str, ...]) -> Self:
        return cls(path, _to_mod(path, roots))


def _args(node: ast.AsyncFunctionDef | ast.FunctionDef) -> Generator[ast.arg]:
    for subnode in ast.walk(node.args):
        if isinstance(subnode, ast.arg):
            yield subnode


def _is_abstract(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute) and node.attr == 'abstractmethod' or
        isinstance(node, ast.Name) and node.id == 'abstractmethod'
    )


class FindUntyped(ast.NodeVisitor):
    def __init__(self) -> None:
        self._mod: list[Mod] = []
        self._stack: list[str] = []
        self._in_func = [False]
        self.potential: list[tuple[Mod, str]] = []

    def visit_module(self, mod: Mod, tree: ast.AST) -> None:
        self._mod.append(mod)
        self.generic_visit(tree)
        self._mod.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(
            self,
            node: ast.AsyncFunctionDef | ast.FunctionDef,
    ) -> None:
        if (
                not self._in_func[-1] and
                not any(_is_abstract(dec) for dec in node.decorator_list)
        ):
            args = tuple(_args(node))
            if node.name == '__init__' and len(args) > 1:
                missing_annotation = False
            else:
                missing_annotation = node.returns is None
            for arg in args:
                if arg.arg not in {'cls', 'self'} and arg.annotation is None:
                    missing_annotation = True
                    break
            if missing_annotation:
                self.potential.append((
                    self._mod[-1], '.'.join((*self._stack, node.name)),
                ))

        self._in_func.append(True)
        self.generic_visit(node)
        self._in_func.pop()

    visit_AsyncFunctionDef = visit_FunctionDef


class _Sig(TypedDict):
    arg_types: list[str]
    return_type: str


class _Ann(TypedDict):
    func_name: str
    line: int
    path: str
    samples: int
    signature: _Sig


class Sig(NamedTuple):
    args: tuple[str, ...]
    ret: str

    @classmethod
    def from_suggestion(cls, suggestion: _Ann) -> Self:
        return cls(
            args=tuple(suggestion['signature']['arg_types']),
            ret=suggestion['signature']['return_type'],
        )


def _suggestions(names: list[tuple[Mod, str]]) -> dict[Mod, dict[int, Sig]]:
    ret: dict[Mod, dict[int, Sig]] = collections.defaultdict(dict)
    for mod, name in names:
        try:
            out = subprocess.check_output((
                sys.executable, '-m', 'mypy.dmypy', 'suggest',
                f'{mod.modname}.{name}', '--no-errors', '--json',
            ))
        except subprocess.CalledProcessError:
            print(f'skipping {name}...')
        else:
            suggestion, = json.loads(out)
            ret[mod][suggestion['line']] = Sig.from_suggestion(suggestion)
    return ret


def _has_any(s: str) -> bool:
    return bool(re.search(r'\bAny\b', s))


def _imports_datetime_module(s: str) -> bool:
    return any(
        isinstance(node, ast.Import) and
        any(name.name == 'datetime' and not name.asname for name in node.names)
        for node in ast.parse(s).body
    )


def _pyupgrade_annotations(tp: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 't.py')

        with open(fname, 'w') as f:
            f.write(f'{_PYUPGRADE_PREFIX}{tp}')

        cmd = (sys.executable, '-mpyupgrade', fname, '--py310-plus')
        subprocess.call(cmd, stderr=subprocess.DEVNULL)

        with open(fname) as f:
            src = f.read()

    assert src.startswith(_PYUPGRADE_PREFIX), src
    return src.removeprefix(_PYUPGRADE_PREFIX)


def _fixup_type(
        modname: str,
        tp: str,
        *,
        imports_datetime_module: bool,
) -> tuple[str, set[str]]:
    imports = set()

    def _symbol_and_name(mod: str, sym: str) -> str:
        if mod == modname:
            # hope there aren't more like these python/mypy#18935
            if sym == 'datetime' and imports_datetime_module:
                return 'datetime.datetime'
            else:
                return sym
        # pyupgrade will fix these
        elif (
                mod == 'typing' and
                sym in {'List', 'Type', 'Tuple', 'Dict', 'Set', 'FrozenSet'}
        ):
            return f'{mod}.{sym}'
        else:
            imports.add(f'from {mod} import {sym.split(".")[0]}')
            return sym

    def cb(m: re.Match[str]) -> str:
        if m[2]:
            return _symbol_and_name(m[1], m[2])
        elif '.' not in m[1]:
            return m[1]
        else:
            return _symbol_and_name(*m[1].rsplit('.', 1))

    tp = re.sub(r'(\w+(?:\.\w+)*)(?::(\w+(?:\.\w+)*))?', cb, tp)

    # convert to new-style annotations
    tp = _pyupgrade_annotations(tp)

    return tp, imports


def _replace(
        i: int,
        tokens: list[Token],
        mod: Mod,
        sig: Sig,
        imports_to_add: set[str],
        *,
        imports_datetime_module: bool,
) -> None:
    for j in range(i, len(tokens)):
        if tokens[j].matches(name='OP', src='('):
            break
    else:
        raise AssertionError('past end?')

    j += 1
    looking_for_name = True
    depth = 1
    names = []
    has_annotation = set()

    while depth:
        token = tokens[j]
        if token.name == 'OP' and token.src in '([{':
            depth += 1
        elif token.name == 'OP' and token.src in ')]}':
            depth -= 1
        elif depth == 1:
            if looking_for_name:
                if token.name == 'NAME':
                    looking_for_name = False
                    names.append(j)
            elif token.matches(name='OP', src=','):
                looking_for_name = True
            elif token.matches(name='OP', src=':'):
                has_annotation.add(names[-1])

        j += 1

    close_paren = j - 1

    for j in range(j, len(tokens)):
        if tokens[j].matches(name='OP', src=':'):
            found_return = False
            break
        elif tokens[j].matches(name='OP', src='->'):
            found_return = True
            break
    else:
        raise AssertionError('past end?')

    if not found_return and not _has_any(sig.ret):
        ann, new_imports = _fixup_type(
            mod.modname,
            sig.ret,
            imports_datetime_module=imports_datetime_module,
        )
        imports_to_add.update(new_imports)
        tokens.insert(close_paren + 1, Token(name='CODE', src=f' -> {ann}'))

    for pos, argtype in zip(reversed(names), reversed(sig.args)):
        if pos not in has_annotation and not _has_any(argtype):
            ann, new_imports = _fixup_type(
                mod.modname,
                argtype,
                imports_datetime_module=imports_datetime_module,
            )
            imports_to_add.update(new_imports)
            tokens.insert(pos + 1, Token(name='CODE', src=f': {ann}'))


def _add_imports(s: str, imports: set[str]) -> str:
    if not imports:
        return s

    sorted_imports = '\n'.join(sorted(imports)) + '\n'
    tree = ast.parse(s)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            continue
        elif (
                isinstance(node, ast.Expr) and
                isinstance(node.value, ast.Constant) and
                isinstance(node.value.value, str)
        ):
            continue  # docstring!
        else:
            lines = s.splitlines(True)
            lines.insert(node.lineno - 1, sorted_imports)
            return ''.join(lines)
    else:
        return f'{s.rstrip()}\n{sorted_imports}'


def _rewrite_src(src: str, mod: Mod, sigs: dict[int, Sig]) -> str:
    imports_datetime_module = _imports_datetime_module(src)

    tokens = src_to_tokens(src)

    imports_to_add: set[str] = set()

    for i, token in reversed_enumerate(tokens):
        if token.matches(name='NAME', src='def') and token.line in sigs:
            _replace(
                i, tokens,
                mod,
                sigs[token.line],
                imports_to_add,
                imports_datetime_module=imports_datetime_module,
            )

    return _add_imports(tokens_to_src(tokens), imports_to_add)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', nargs='+')
    parser.add_argument(
        '--application-directories', default='.',
        help='colon-separated app roots, such as `.:src`, default: `.`',
    )
    args = parser.parse_args(argv)

    roots = tuple(args.application_directories.split(':'))

    visitor = FindUntyped()
    for fname in args.fname:
        with open(fname, 'rb') as fb:
            tree = ast.parse(fb.read(), filename=fname)
        visitor.visit_module(Mod.from_path(fname, roots), tree)

    for mod, sigs in _suggestions(visitor.potential).items():
        with open(mod.path, encoding='UTF-8') as f:
            src = _rewrite_src(f.read(), mod, sigs)
        with open(mod.path, 'w', encoding='UTF-8') as f:
            f.write(src)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
