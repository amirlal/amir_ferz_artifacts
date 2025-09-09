# audit_corpus.py
import json
import math
import argparse
import csv
from collections import Counter
from selden_vault import SeldenVault
vault = SeldenVault(protocol_name="EntropyAudit", steward="Amir")


def compute_entropy(tokens):
    counts = Counter(tokens)
    total = sum(counts.values())
    probs = [count / total for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def kl_divergence(p_dist, q_dist):
    return sum(p * math.log2(p / q) for p, q in zip(p_dist, q_dist) if p > 0 and q > 0)

def baseline_distribution(tokens):
    unique = set(tokens)
    return [1 / len(unique)] * len(unique)

def audit_file(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = [json.loads(line)['text'] for line in f]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['line_number', 'entropy', 'kl_divergence'])

        for i, line in enumerate(lines):
            tokens = line.split()
            entropy = compute_entropy(tokens)
            baseline = baseline_distribution(tokens)
            actual_dist = [tokens.count(t)/len(tokens) for t in set(tokens)]
            kl = kl_divergence(actual_dist, baseline)
            writer.writerow([i + 1, round(entropy, 4), round(kl, 4)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='input_path', required=True)
    parser.add_argument('--out', dest='output_path', required=True)
    args = parser.parse_args()
    audit_file(args.input_path, args.output_path)
