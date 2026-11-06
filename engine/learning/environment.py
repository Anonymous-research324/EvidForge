from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
RUNTIME=ROOT/'source_runtime'
sys.path.insert(0,str(RUNTIME));sys.path.insert(0,str(ROOT))
