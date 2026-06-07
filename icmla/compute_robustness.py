"""
Program to read results_all_2000_samples.txt and identify high-risk driving scenarios.

python icmla/compute_robustness.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

def parse_results_file(filepath: str) -> List[Tuple[int, str, Dict]]:
    """
    Parse the results file and extract sample info, paths, and JSON responses.
    
    Returns:
        List of tuples: (sample_number, image_path, response_dict)
    """
    results = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by Sample entries
    sample_blocks = re.split(r'Sample \d+:', content)
    
    for idx, block in enumerate(sample_blocks[1:], 1):  # Skip the header
        try:
            # Extract image path
            path_match = re.search(r'Path: (.+?)\n', block)
            if not path_match:
                continue
            
            image_path = path_match.group(1).strip()
            
            # Extract the JSON response (it comes after "assistant" keyword)
            # Look for the last JSON object in the block (the actual response, not the prompt template)
            json_objects = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', block, re.DOTALL))
            
            if not json_objects:
                continue
            
            # Get the last JSON object (which should be the actual response)
            json_match = json_objects[-1]
            json_str = json_match.group(0)
            
            # Clean up the JSON string - replace float placeholders with actual numbers
            # and fix common JSON issues
            json_str = json_str.replace(': float,', ': 0.0,')
            json_str = json_str.replace(': float}', ': 0.0}')
            json_str = json_str.replace(': string,', ': "N/A",')
            json_str = json_str.replace(': string}', ': "N/A"}')
            
            response_dict = json.loads(json_str)
            
            results.append((idx, image_path, response_dict))
            
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            print(f"Warning: Failed to parse sample {idx}: {e}")
            continue
    
    return results


from collections import Counter

ACTION_RANK = {
    "stop":0,
    "slow_down":1,
    "maintain":2,
    "accelerate":3,
    "uncertain":4,
}


def calculate_metrics(baseline_results, attack_results):

    # path -> action
    baseline = {
        path: response["ego_action"]
        for _, path, response in baseline_results
    }

    attack = {
        path: response["ego_action"]
        for _, path, response in attack_results
    }

    common_paths = baseline.keys() & attack.keys()

    if not common_paths:
        raise ValueError("No matching paths found")

    total = len(common_paths)

    changed = 0
    safety_regressions = 0
    uncertainty_induced = 0

    ads_sum = 0.0
    ads_count = 0

    pss_sum = 0.0

    transition_counter = Counter()

    for path in common_paths:

        base_action = baseline[path]
        attack_action = attack[path]

        transition_counter[(base_action, attack_action)] += 1

        # PSS
        if base_action == attack_action:
            pss_sum += 1.0
        else:
            pss_sum += 0.5
            changed += 1

        # uncertainty
        if (
            base_action != "uncertain"
            and attack_action == "uncertain"
        ):
            uncertainty_induced += 1

        # ADS / SRR
        if (
            base_action in ACTION_RANK
            and attack_action in ACTION_RANK
        ):
            ads_sum += abs(
                ACTION_RANK[base_action]
                - ACTION_RANK[attack_action]
            )

            ads_count += 1

            if (
                ACTION_RANK[attack_action]
                > ACTION_RANK[base_action]
            ):
                safety_regressions += 1

    metrics = {
        "matched_samples": total,
        "DCR": changed / total,
        "ADS": ads_sum / ads_count if ads_count else 0.0,
        "SRR": safety_regressions / total,
        "PSS": pss_sum / total,
        "UIR": uncertainty_induced / total,
        "transition_counts": transition_counter,
    }

    return metrics

if __name__ == "__main__":
    results_file = "/Users/ethangao/icmla/inference_results_exp_counterfactual/results_7_8_samples_p4_add_ped_crossing.txt"
    # output_file = "/Users/ethangao/icmla/high_risk_report.txt"
    
    print("Reading results file...")
    baseline_results = parse_results_file(results_file)
    # print(f"✓ Parsed res: {results} ")
    for idx, path, res in baseline_results:
        print(f"Sample {idx}: Path: {path}, Response: {res}\n")
    print(f"✓ Parsed {len(baseline_results)} samples")
    
    results_file2 = "/Users/ethangao/icmla/inference_results_exp_counterfactual/results_7_8_samples_p4.txt"
    attack_results = parse_results_file(results_file2)
    # print(f"✓ Parsed res: {results} ")
    for idx, path, res in attack_results:
        print(f"Sample {idx}: Path: {path}, Response: {res}\n")
    print(f"✓ Parsed {len(attack_results)} samples")
    
    metrics = calculate_metrics(
        baseline_results,
        attack_results
    )

    for k, v in metrics.items():
        if k != "transition_counts":
            print(f"{k}: {v}")