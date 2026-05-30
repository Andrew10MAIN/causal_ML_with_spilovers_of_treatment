import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  
sys.path.append(str(BASE_DIR))


from pipelines.generate_data import main as generate_data
from pipelines.run_models import main as run_models

def main():
    print("=== GENERATING DATA ===")
    generate_data()

    print("=== RUNNING MODELS ===")
    run_models()

if __name__ == "__main__":
    main()