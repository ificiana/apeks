import sys
from pathlib import Path

# Add src directory to module search path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from apeks.app import main

if __name__ == "__main__":
    main()
