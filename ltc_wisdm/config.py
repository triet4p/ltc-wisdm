from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RAW_DATA_FILE = RAW_DATA_DIR / 'WISDM_ar_v1.1_raw.txt'

WINDOW_SIZE = 80
STEP_SIZE = 40

RAW_DATA_COL_NAMES = ['user', 'activity', 'timestamp', 'x', 'y', 'z']