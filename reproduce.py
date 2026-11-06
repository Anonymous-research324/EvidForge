#!/usr/bin/env python3
"""Paper artifact CLI. Expected metrics are never consumed by experiment workers."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"experiments"))
from release_runtime import main
if __name__=="__main__":main()
