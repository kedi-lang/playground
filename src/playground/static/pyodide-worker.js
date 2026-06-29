import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";

const pyodidePromise = loadPyodide();

const DRIVER = String.raw`
import builtins
import contextlib
import io
import json
import traceback
import types
import typing


class _KediAttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _from_json(value):
    if isinstance(value, dict):
        return _KediAttrDict({key: _from_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_from_json(item) for item in value]
    return value


def _to_json(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Pyodide mappings require string keys")
        return {key: _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_json(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, (type, types.ModuleType)):
        return {
            key: _to_json(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    raise TypeError(f"Python value {type(value).__name__} cannot cross the Pyodide bridge")


def _type_descriptor(value):
    if value is type(None):
        return {"kind": "none"}
    if isinstance(value, type) and value.__module__ == "builtins":
        return {"kind": "builtin", "name": value.__qualname__}
    if value is typing.Any:
        return {"kind": "typing", "name": "Any"}

    origin = typing.get_origin(value)
    args = typing.get_args(value)
    if origin in (typing.Union, types.UnionType):
        return {"kind": "union", "args": [_type_descriptor(arg) for arg in args]}
    if origin is typing.Literal:
        return {"kind": "literal", "values": [_to_json(arg) for arg in args]}
    if origin is not None:
        origin_descriptor = _type_descriptor(origin)
        if not args:
            return origin_descriptor
        return {
            "kind": "generic",
            "origin": origin_descriptor,
            "args": [_type_descriptor(arg) for arg in args],
        }
    raise TypeError("Python-defined types cannot cross the Pyodide bridge yet")


def _run_block(namespace, code, sync_names):
    header = ["def __kedi_wrapper__():"]
    if sync_names:
        header.append(f"    global {', '.join(sync_names)}")
    body = [f"    {line}" for line in code.splitlines()]
    source = "\n".join(header + body + ["    pass"])
    exec(compile(source, "<kedi-pyodide:block>", "exec"), namespace, namespace)
    wrapper = namespace.pop("__kedi_wrapper__")
    return wrapper()


def _serializable_namespace(namespace, names):
    result = {}
    for name in names:
        if name.startswith("_") or name not in namespace:
            continue
        try:
            result[name] = _to_json(namespace[name])
        except TypeError:
            continue
    return result


def _execute_kedi_request(request):
    namespace = globals().setdefault(
        "__kedi_user_namespace",
        {"__builtins__": builtins.__dict__},
    )
    namespace.update(
        {name: _from_json(value) for name, value in request.get("env", {}).items()}
    )
    action = request["action"]
    code = request["code"]
    sync_names = request.get("syncNames", [])

    if action == "evaluate_inline":
        try:
            result = eval(compile(code, "<kedi-pyodide:inline>", "eval"), namespace, namespace)
        except SyntaxError:
            _run_block(namespace, code, sync_names)
            result = ""
    elif action == "execute_block":
        result = _run_block(namespace, code, sync_names)
    elif action == "execute_side_effects":
        exec(compile(code, "<kedi-pyodide:side-effects>", "exec"), namespace, namespace)
        result = None
    elif action == "execute_prelude":
        exec(compile(code, "<kedi-pyodide:prelude>", "exec"), namespace, namespace)
        result = None
    elif action == "evaluate_type_expression":
        value = eval(compile(code, "<kedi-pyodide:type>", "eval"), namespace, namespace)
        return {"ok": True, "type": _type_descriptor(value), "env": {}}
    else:
        raise ValueError(f"Unknown Pyodide action: {action!r}")

    names = (
        [name for name in namespace if not name.startswith("_")]
        if action == "execute_prelude"
        else sync_names
    )
    return {
        "ok": True,
        "result": _to_json(result),
        "env": _serializable_namespace(namespace, names),
    }


__kedi_stdout = io.StringIO()
__kedi_stderr = io.StringIO()
with contextlib.redirect_stdout(__kedi_stdout), contextlib.redirect_stderr(__kedi_stderr):
    try:
        __kedi_response = _execute_kedi_request(json.loads(__kedi_request_json))
    except Exception as exc:
        __kedi_response = {
            "ok": False,
            "errorType": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

__kedi_response["stdout"] = __kedi_stdout.getvalue()
__kedi_response["stderr"] = __kedi_stderr.getvalue()

json.dumps(__kedi_response)
`;

self.addEventListener("message", async (event) => {
  const { id, request } = event.data;
  try {
    const pyodide = await pyodidePromise;
    pyodide.globals.set("__kedi_request_json", JSON.stringify(request));
    const responseJson = await pyodide.runPythonAsync(DRIVER);
    self.postMessage({ id, response: JSON.parse(responseJson) });
  } catch (error) {
    self.postMessage({
      id,
      response: {
        ok: false,
        errorType: error?.name || "PyodideError",
        error: error?.message || String(error),
      },
    });
  }
});
