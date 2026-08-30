import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from tree.tree import load_tree
from engine import outcomes


class Workspace:
    def __init__(self, name: str):
        self.dir  = ROOT / 'workspaces' / name
        tree      = load_tree(self.dir / 'universe.json')
        meta      = tree['meta']

        self.horizons     = meta['horizons']
        self.start_date   = meta['start_date']
        self.asset        = meta['asset']
        self.n_thresholds = meta.get('n_thresholds', 30)

        interval = self.asset.get('interval', '1d')
        self.horizon_unit = 'h' if interval.endswith('h') else 'd'

        self.sample_freq  = meta['sample_freq']
        self.min_obs      = meta.get('min_obs', 100)
        self.read_min_dev = meta.get('read_min_dev', 10.0)
        self.read_min_n   = meta.get('read_min_n', 50)
        self._display_nums = meta['display_horizons']

        self._load_plugin()

        # The outcome is loaded after the plugin so a workspace can register its own.
        oc = meta.get('outcome', {'name': 'up', 'params': {}})
        self.set_outcome(oc.get('name', 'up'), oc.get('params') or {})

    def set_outcome(self, name: str, params: dict | None = None) -> None:
        """Point the workspace at a different outcome (used by --outcome)."""
        self.outcome        = name
        self.outcome_params = params or {}
        self.outcome_spec   = outcomes.get(name)
        self.event          = outcomes.event_label(name, self.outcome_params)
        self.outcome_expr   = outcomes.describe(name, self.outcome_params)

    @property
    def is_default_outcome(self) -> bool:
        return self.outcome == 'up'

    def _load_plugin(self) -> None:
        plugin_path = self.dir / 'plugin.py'
        if not plugin_path.exists():
            return
        spec = importlib.util.spec_from_file_location(
            f'_ws_plugin_{self.dir.name}', plugin_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    @property
    def tree_path(self) -> Path:
        return self.dir / 'universe.json'

    @property
    def log(self) -> Path:
        return self.dir / 'log.jsonl'

    def output_dir(self, family: str) -> Path:
        return self.dir / family

    def output_path(self, family: str, node_id: str) -> Path:
        """
        Where a node's xlsx lives. The default 'up' outcome keeps the historical
        flat path so the 65 existing files stay in place; every other outcome gets
        its own subdirectory, so outcomes never overwrite one another.
        """
        if self.is_default_outcome:
            return self.dir / family / f'{node_id}.xlsx'
        return self.dir / family / self.event / f'{node_id}.xlsx'

    def family_log(self, family: str) -> Path:
        return self.dir / family / 'log.jsonl'

    @property
    def key_horizons(self) -> list[str]:
        return [f'+{h}{self.horizon_unit}' for h in self.horizons]

    @property
    def pivot_horizon(self) -> str:
        h = self.horizons
        return f'+{h[(len(h) - 1) // 2]}{self.horizon_unit}'

    @property
    def display_horizons(self) -> list[str]:
        return [f'+{n}{self.horizon_unit}' for n in self._display_nums]
