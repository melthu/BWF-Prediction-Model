import subprocess
import sys
import time

STEPS = [
    ("Building 5-year tournament config",      "src/build_config.py"),
    ("Scraping Wikipedia match results",        "src/scraper_orchestrator.py"),
    ("Engineering temporal features",           "src/feature_engineering.py"),
    ("Mirroring dataset for ML readiness",      "src/data_loader.py"),
]


def run_pipeline():
    print("=" * 60)
    print("  BWF Men's Singles — End-to-End Data Pipeline")
    print("=" * 60)

    pipeline_start = time.time()

    for i, (label, script) in enumerate(STEPS, start=1):
        print(f"\n[{i}/{len(STEPS)}] {label}...")
        step_start = time.time()

        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,   # let stdout/stderr stream to the terminal
        )

        elapsed = time.time() - step_start

        if result.returncode != 0:
            print(f"\n[ERROR] Step {i} failed (exit code {result.returncode}). Pipeline halted.")
            sys.exit(result.returncode)

        print(f"[{i}/{len(STEPS)}] Done in {elapsed:.1f}s")

    total = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print(f"  Pipeline complete! Total time: {total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
