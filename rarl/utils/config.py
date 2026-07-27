from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return _minimal_yaml_load(text)


def _minimal_yaml_load(text):
    root = {}
    stack = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, raw_value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip() == "":
            value = {}
            parent[key] = value
            stack.append((indent, value))
        else:
            parent[key] = _parse_scalar(raw_value.strip())
    return root


def _parse_scalar(value):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
