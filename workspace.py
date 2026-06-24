import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from tree.tree import load_tree


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
