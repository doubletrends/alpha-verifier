from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'pipeline' / 'output'

ODS_CSV = DATA / 'step1_ods.csv'
DWD_CSV = DATA / 'step2_dwd.csv'


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
