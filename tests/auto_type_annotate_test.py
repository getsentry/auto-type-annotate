from __future__ import annotations

import ast
import contextlib
import ntpath
import os
import subprocess
import sys
from unittest.mock import patch

from auto_type_annotate import _add_imports
from auto_type_annotate import _rewrite_src
from auto_type_annotate import _suggestions
from auto_type_annotate import _to_mod
from auto_type_annotate import FindUntyped
from auto_type_annotate import main
from auto_type_annotate import Mod
from auto_type_annotate import Sig

_MOD = Mod('t.py', 't')


def test_to_mod_uses_longest_path():
    assert _to_mod('a/b/c.py', ('.', 'a')) == 'b.c'
    assert _to_mod('a/b/c.py', ('a', '.')) == 'b.c'


@patch.object(os.path, 'relpath', new=ntpath.relpath)
def test_to_mod_uses_longest_path_windows():
    assert _to_mod(r'a\b\c.py', ('.', 'a')) == 'b.c'


def test_to_mod_src_layout():
    assert _to_mod('src/foo.py', ('.', 'src')) == 'foo'
    assert _to_mod('bar.py', ('.', 'src')) == 'bar'


def _find_untyped(s):
    visitor = FindUntyped()
    visitor.visit_module(_MOD, ast.parse(s))
    return visitor.potential


def test_find_untyped_trivial():
    assert _find_untyped('') == []


def test_find_untyped_all_things_typed():
    src = '''\
def f() -> int: ...
def g(x: int) -> None: ...
class C:
    def f(self, x: int) -> None: ...
    @classmethod
    def g(cls, x: int) -> None: ...
'''
    assert _find_untyped(src) == []


def test_find_untyped_init_with_no_args_needs_a_return_annotation():
    src = '''\
class C:
    def __init__(self): ...
'''
    assert _find_untyped(src) == [(_MOD, 'C.__init__')]


def test_find_untyped_init_does_not_need_a_return_value():
    src = '''\
class C:
    def __init__(self, x: int): ...
'''
    assert _find_untyped(src) == []


def test_find_untyped_missing_argument_annotation():
    src = 'def f(x) -> None: ...'
    assert _find_untyped(src) == [(_MOD, 'f')]


def test_find_untyped_missing_return_annotation():
    src = 'def f(x: int): ...'
    assert _find_untyped(src) == [(_MOD, 'f')]


def test_find_untyped_ignores_nested_functions():
    src = '''\
def f() -> None:
    def g():  # technically untyped but dmypy can't help
        print('hello hello world')
    g()
'''
    assert _find_untyped(src) == []


def test_find_untyped_async_def():
    src = 'async def f(): ...'
    assert _find_untyped(src) == [(_MOD, 'f')]


def test_find_untyped_skips_abstract():
    src = '''\
import abc

class C:
    @abc.abstractmethod
    def f(self): pass
'''
    assert _find_untyped(src) == []


def test_find_untyped_skips_abstract_from_imported():
    src = '''\
from abc import abstractmethod

class C:
    @abstractmethod
    def f(self): pass
'''
    assert _find_untyped(src) == []


@contextlib.contextmanager
def _dmypy():
    subprocess.check_call((sys.executable, '-m', 'mypy.dmypy', 'run', '.'))
    try:
        yield
    finally:
        subprocess.check_call((sys.executable, '-m', 'mypy.dmypy', 'stop'))


def test_suggestions(tmp_path):
    src = '''\
def unknown_dec(f):
    return f

def f():
    print('hello world')

@unknown_dec  # dmypy won't give us something for this!
def g():
    print('hello world')
'''
    tmp_path.joinpath('t.py').write_text(src)

    with contextlib.chdir(tmp_path), _dmypy():
        ret = _suggestions([(_MOD, 'unknown_dec'), (_MOD, 'f'), (_MOD, 'g')])

    assert ret == {
        _MOD: {
            1: Sig(args=('Callable[[], Any]',), ret='Callable[[], Any]'),
            4: Sig(args=(), ret='None'),
        },
    }


def test_add_imports_trivial():
    assert _add_imports('', set()) == ''


def test_add_imports_before_first_non_future_import():
    src = '''\
from __future__ import annotations

import os
'''
    expected = '''\
from __future__ import annotations

from baz import womp
from foo import bar
import os
'''
    imports = {'from foo import bar', 'from baz import womp'}
    assert _add_imports(src, imports) == expected


def test_add_imports_body_is_only_future_import():
    src = 'from __future__ import annotations'
    expected = '''\
from __future__ import annotations
from foo import bar
'''
    assert _add_imports(src, {'from foo import bar'}) == expected


def test_add_imports_to_trivial_body():
    src = '''\
# comment!
'''
    expected = '''\
# comment!
from foo import bar
'''
    assert _add_imports(src, {'from foo import bar'}) == expected


def test_add_imports_with_docstring():
    src = '''\
"""hello world"""
from __future__ import annotations
'''
    expected = '''\
"""hello world"""
from __future__ import annotations
from foo import bar
'''
    assert _add_imports(src, {'from foo import bar'}) == expected


def test_rewrite_src_simple():
    src = '''\
def f(x):
    return f'hello hello {x}'

f('anthony')
'''
    expected = '''\
def f(x: str) -> str:
    return f'hello hello {x}'

f('anthony')
'''
    sigs = {1: Sig(args=('str',), ret='str')}

    assert _rewrite_src(src, _MOD, sigs) == expected


def test_avoids_dmypy_datetime_bug():
    # python/mypy#18935
    src = '''\
import datetime

def f(x):
    return x.isoformat()

f(datetime.datetime.now())
'''
    expected = '''\
import datetime

def f(x: datetime.datetime) -> str:
    return x.isoformat()

f(datetime.datetime.now())
'''

    sigs = {3: Sig(args=('t.datetime',), ret='str')}

    assert _rewrite_src(src, _MOD, sigs) == expected


def test_imported_name_in_module():
    src = '''\
from collections import deque

def f(d):
    return d.pop()

f(deque([1, 2, 3]))
'''
    expected = '''\
from collections import deque

def f(d: deque[int]) -> int:
    return d.pop()

f(deque([1, 2, 3]))
'''

    sigs = {3: Sig(args=('t.deque[int]',), ret='int')}

    assert _rewrite_src(src, _MOD, sigs) == expected


def test_imported_pep_585_type_name():
    src = '''\
class C: pass

def f():
    return [C()]
'''
    expected = '''\
class C: pass

def f() -> list[C]:
    return [C()]
'''
    sigs = {3: Sig(args=(), ret='typing:List[t.C]')}

    assert _rewrite_src(src, _MOD, sigs) == expected


def test_needs_to_add_imports():
    src = '''\
from b import C

def f():
    return C().f()
'''
    expected = '''\
from b import D
from b import C

def f() -> D:
    return C().f()
'''
    sigs = {3: Sig(args=(), ret='b.D')}

    assert _rewrite_src(src, _MOD, sigs) == expected


def test_return_already_annotated():
    src = '''\
def f(x) -> int:
    return 2 * x

f(1)
'''
    expected = '''\
def f(x: int) -> int:
    return 2 * x

f(1)
'''
    sigs = {1: Sig(args=('int',), ret='int')}

    assert _rewrite_src(src, _MOD, sigs) == expected


def test_argument_already_annotated():
    src = '''\
def f(x: list[int], y):
    return sum(x) + y

f([1, 2, 3], 4)
'''
    expected = '''\
def f(x: list[int], y: int) -> int:
    return sum(x) + y

f([1, 2, 3], 4)
'''
    sigs = {1: Sig(args=('typing.List[int]', 'int'), ret='int')}

    assert _rewrite_src(src, _MOD, sigs) == expected


def test_Any_argument_is_skipped():
    src = '''\
def f(x):
    return str(x)
'''
    expected = '''\
def f(x) -> str:
    return str(x)
'''
    sigs = {1: Sig(args=('Any',), ret='str')}

    assert _rewrite_src(src, _MOD, sigs) == expected


def test_Any_return_annotation_is_skipped():
    src = '''\
def f(x, y):
    if x > 0:
        return y[0]
    else:
        return y[1]


def g(x: int, y):
    f(x, y)
'''
    expected = '''\
def f(x: int, y):
    if x > 0:
        return y[0]
    else:
        return y[1]


def g(x: int, y):
    f(x, y)
'''
    sigs = {1: Sig(args=('int', 'Any'), ret='Any')}

    assert _rewrite_src(src, _MOD, sigs) == expected


def test_integration(tmp_path):
    src = '''\
def f():
    print('hello hello world')
'''
    expected = '''\
def f() -> None:
    print('hello hello world')
'''

    t_py = tmp_path.joinpath('t.py')
    t_py.write_text(src)
    with contextlib.chdir(tmp_path), _dmypy():
        assert not main(('t.py',))
    assert t_py.read_text() == expected


def test_integration_src_path(tmp_path):
    src = '''\
class C: pass

def f():
    return C()
'''
    expected = '''\
class C: pass

def f() -> C:
    return C()
'''

    tmp_path.joinpath('src').mkdir()
    t_py = tmp_path.joinpath('src/t.py')
    t_py.write_text(src)

    with contextlib.chdir(tmp_path), _dmypy():
        assert not main(('src/t.py', '--application-directories=.:src'))
    assert t_py.read_text() == expected
