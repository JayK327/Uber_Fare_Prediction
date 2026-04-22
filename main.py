"""main.py — run full pipeline: python main.py [--no-tune]"""
import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))


def main(tune=True):
    t0 = time.time()

    print("\n=== STEP 1: Generate / Load Dataset ===")
    if not os.path.exists("data/raw/uber_fares.csv"):
        from src.generate_dataset import generate
        generate()
    else:
        print("Found existing data/raw/uber_fares.csv — skipping generation.")

    print("\n=== STEP 2: Data Cleaning ===")
    from src.data_cleaning import run as clean_run
    clean_run()

    print("\n=== STEP 3: Feature Engineering ===")
    from src.feature_engineering import run as feat_run
    feat_run()

    print("\n=== STEP 4: Model Training ===")
    from src.model_training import run as model_run
    model_run(tune=tune)

    print(f"\n✓ Pipeline complete in {(time.time()-t0)/60:.1f} min")
    print("  models/          — saved .joblib files")
    print("  outputs/plots/   — charts")
    print("  outputs/model_results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-tune", action="store_true")
    args = p.parse_args()
    main(tune=not args.no_tune)
