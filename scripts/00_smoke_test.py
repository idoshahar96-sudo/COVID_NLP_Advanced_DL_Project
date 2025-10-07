# Make 'src' importable without changing environment variables
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from covid_nlp.utils import setup_logging, set_seed, hello

def main():
    setup_logging()
    set_seed(123)
    print(hello())

if __name__ == "__main__":
    main()
