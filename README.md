[![build status](https://github.com/getsentry/auto-type-annotate/actions/workflows/main.yml/badge.svg)](https://github.com/getsentry/auto-type-annotate/actions/workflows/main.yml)

auto-type-annotate
==================

automatically add annotations to untyped python code!

## installation

```bash
pip install auto-type-annotate
```

## usage

this tool is intented to help you automatically add annotations while
gradually typing a code base!

there's a few pre-requisites you'll need to do in order to use the tool
successfully

first you'll need to make your codebase pass `mypy` with the following settings
enabled:

- `check_untyped_defs = true`: this is needed to analyze types in functions
  we'd like to automatically type
- `local_partial_types = true`: this is a setting forced by `dmypy` -- which
  we'll be using to generate typing suggestions

you can check whether you're "ready" to use this by running the following:

```bash
dmypy stop
dmypy run
```

it should look something like:

```console
$ dmypy stop
Daemon stopped
$ dmypy run
Daemon started
Success: no issues found in 6839 source files
```

once that passes -- and `dmypy` is still running: use the tool on whatever
files you'd like to improve!

for instance after:

```bash
auto-type-annotate \
    --application-directories .:src \
    src/sentry/api/authentication.py
```

```diff
$ git diff
diff --git a/src/sentry/api/authentication.py b/src/sentry/api/authentication.py
index ec0526b9f62..ac797d136f5 100644
--- a/src/sentry/api/authentication.py
+++ b/src/sentry/api/authentication.py
@@ -86,7 +86,7 @@ class AuthenticationSiloLimit(SiloLimit):
         )


-def is_internal_relay(request, public_key):
+def is_internal_relay(request: Request, public_key: str) -> bool:
     """
     Checks if the relay is trusted (authorized for all project configs)
     """
@@ -99,7 +99,7 @@ def is_internal_relay(request, public_key):
     return is_internal_ip(request)


-def is_static_relay(request):
+def is_static_relay(request: Request) -> bool:
     """
     Checks if the request comes from a statically configured relay

@@ -141,7 +141,7 @@ def relay_from_id(request: Request, relay_id: str) -> tuple[Relay | None, bool]:
             return None, False  # no Relay found


```

## why not pyannotate?

:'(

```console
$ pyannotate --help
Traceback (most recent call last):
  File ".venv/bin/pyannotate", line 5, in <module>
    from pyannotate_tools.annotations.__main__ import main
  File ".venv/lib/python3.13/site-packages/pyannotate_tools/annotations/__main__.py", line 9, in <module>
    from lib2to3.main import StdoutRefactoringTool
ModuleNotFoundError: No module named 'lib2to3'
```
