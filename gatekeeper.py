import subprocess
import sys
import os

def run_gatekeeper():
    print("Running Pipeline Gatekeeper...")
    
    cmd = [
        "python3", "utils/validate_submission.py",
        "--matching", "output/matching_results.tsv",
        "--candidate", "output/candidate_pairs.tsv",
        "--test-dir", "dataset/test"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0 or "PASS" not in result.stdout:
        print("\n[CRITICAL FAILURE] Validation failed. Bharath's outputs are trash.")
        # Sabotage the bad files so they cannot be accidentally uploaded
        os.rename("output/matching_results.tsv", "output/FAILED_matching_results.tsv")
        os.rename("output/candidate_pairs.tsv", "output/FAILED_candidate_pairs.tsv")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Outputs are bulletproof. Safe to upload to Leaderboard.")

if __name__ == "__main__":
    run_gatekeeper()
    