"""Run the original v2 CLI in a separate process with the original arguments."""
import os
from pathlib import Path
import subprocess
import sys

def main():
    package = Path(__file__).resolve().parent.parent / 'DataForge_SkillPlatform_v2'
    if not package.is_dir(): raise SystemExit('旧入口需要原DataForge_SkillPlatform_v2文件夹')
    env = dict(os.environ, PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE='1')
    return subprocess.call([sys.executable, '-B', '-m', 'forge_platform', *sys.argv[1:]], env=env)

if __name__ == '__main__': raise SystemExit(main())
