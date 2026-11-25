"""Strict, GT-backed pilot rewards. No model-generated code is evaluated."""
import ast
import collections
import importlib.util
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / 'third_party' / '_HypoSpace'
_bool_module = None

def pack(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))

def boolean_key(expression, variables):
    global _bool_module
    expr = expression.strip()
    expr = re.sub(r'\bAND\b', '&', expr)
    expr = re.sub(r'\bOR\b', '|', expr)
    expr = re.sub(r'\bNOT\b', '~', expr)
    tree = ast.parse(expr.strip(), mode='eval')
    permitted = (ast.Expression, ast.Name, ast.Load, ast.UnaryOp, ast.Invert,
                 ast.BinOp, ast.BitAnd, ast.BitOr)
    for node in ast.walk(tree):
        if not isinstance(node, permitted):
            raise ValueError('Outside Boolean grammar')
        if isinstance(node, ast.Name) and node.id not in variables:
            raise ValueError('Unknown variable')
    if _bool_module is None:
        path = BASE/'boolean/boolean_dataset.py'
        spec = importlib.util.spec_from_file_location('official_boolean_dataset', path)
        _bool_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_bool_module)
    b = _bool_module.BooleanExpression(expr, variables, {'AND','OR','NOT'})
    if b.sympy_expr is None:
        raise ValueError('Unparseable')
    return pack(b.mechanistic_key(apply_commutativity=True,
            apply_idempotence_and_or=True, flatten_associativity=True))

def parse_key(text, record):
    text = text.strip()
    domain = record['domain']
    if domain == 'list':
        match = re.fullmatch(r'<answer>\s*([^<>\n]+?)\s*</answer>', text)
        return match.group(1) if match else None
    if domain == 'causal':
        match = re.fullmatch(r'Graph:\s*(.+)', text, re.I)
        if not match:
            return None
        body = match.group(1).strip()
        if body.lower() == 'no edges':
            return pack([])
        edges = []
        for part in body.split(','):
            m = re.fullmatch(r'\s*([A-Za-z0-9_]+)\s*(?:->|→)\s*([A-Za-z0-9_]+)\s*', part)
            if not m or m[1] not in record['variables'] or m[2] not in record['variables']:
                return None
            edges.append((m[1], m[2]))
        return pack(sorted(set(edges)))
    if domain == 'boolean':
        match = re.fullmatch(r'Expression:\s*([^\n]+)', text, re.I)
        if not match:
            return None
        try:
            return boolean_key(match[1], record['variables'])
        except (ValueError, SyntaxError, TypeError, AttributeError, RecursionError):
            return None
    if domain == '3d':
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if not lines or lines.pop(0) != 'Structure:':
            return None
        n = record['grid_size']; layers = []; index = 0
        while index < len(lines):
            if lines[index] != f'Layer {len(layers)+1}:':
                return None
            index += 1; layer = []
            for _ in range(n):
                if index >= len(lines):
                    return None
                row = lines[index].split(); index += 1
                if len(row) != n or any(x not in ('0','1') for x in row):
                    return None
                layer.extend(map(int,row))
            layers.append(layer)
        if not layers or len(layers) > record['max_height']:
            return None
        # Remove only trailing all-zero padding, matching official equivalence.
        while layers and not any(layers[-1]):
            layers.pop()
        return pack(layers) if layers else None
    raise ValueError(domain)

def candidate_key(text, record):
    key = parse_key(text, record)
    return key if key in record['gt_keys'] else None

def group_rewards(keys, num_valid_answers, kind='gapo'):
    if num_valid_answers < 1:
        raise ValueError('Empty GT')
    counts = collections.Counter(k for k in keys if k is not None)
    nv = sum(counts.values())
    if kind == 'validity':
        return [1.0 if k is not None else -1.0 for k in keys]
    if kind != 'gapo':
        raise ValueError(kind)
    return [1-counts[k]/nv+1/num_valid_answers if k is not None else -1.0 for k in keys]

def grouped_reward(records):
    """Global records can arrive interleaved across ranks; caller handles gather/scatter."""
    groups = collections.defaultdict(list); result = {}
    for row in records:
        groups[row['group_id']].append(row)
    for rows in groups.values():
        if len(rows) != rows[0]['G'] or len({r['L'] for r in rows}) != 1:
            raise ValueError('Incomplete or mixed group')
        values = group_rewards([r['key'] for r in rows], rows[0]['L'])
        for r, value in zip(rows,values):
            result[r['index']] = value
    return result
