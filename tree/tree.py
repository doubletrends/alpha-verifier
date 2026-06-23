import json
from pathlib import Path


def load_tree(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def save_tree(tree: dict, path: Path) -> None:
    path.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding='utf-8')


def all_nodes(tree: dict) -> list[dict]:
    return [n for nodes in tree['families'].values() for n in nodes]


def find_node(tree: dict, node_id: str) -> dict:
    for node in all_nodes(tree):
        if node['id'] == node_id:
            return node
    raise ValueError(f"Node '{node_id}' not found in universe.json")


def pending_in_family(tree: dict, family: str) -> list[dict]:
    return [n for n in tree['families'].get(family, []) if n.get('status') == 'pending']


def all_in_family(tree: dict, family: str) -> list[dict]:
    return tree['families'].get(family, [])


def next_pending(tree: dict) -> dict | None:
    for node in all_nodes(tree):
        if node.get('status') == 'pending':
            return node
    return None
