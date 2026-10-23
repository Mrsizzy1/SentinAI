"""
Point d'entrée : redirige vers run.py.
Supporte aussi : python main.py --https  ou  python main.py --desktop
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from run import main

if __name__ == "__main__":
    main()
