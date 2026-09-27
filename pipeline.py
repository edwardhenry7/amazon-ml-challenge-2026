"""
Entity resolution pipeline for the Amazon ML Challenge business-matching task.

Stages:
  1. Load + normalize Source 1 / 2 / 3
  2. Build country-segmented, token-based inverted index over Source 2 + Source 3
  3. For each Source 1 entity, generate candidates via shared tokens (blocking)
  4. Score candidates with fuzzy name/address similarity
  5. Threshold to produce final matches
  6. Write candidate_pairs.tsv and matching_results.tsv
  7. (optional) Evaluate against ground truth: blocking recall + approximate F_0.5

Usage examples
--------------
Quick sample test (recommended FIRST — few minutes):
    python pipeline.py --sample 3000 --eval --out-dir sample_out

Full run (all 2.2M Source 1 entities):
    python pipeline.py --full --out-dir output

You can tune:
    --topk         max candidates kept per S1 entity in candidate_pairs.tsv (default 15)
    --threshold    min score (0-100) to keep a candidate as a final match (default 90)
    --max-posting  skip tokens that appear in more than this many records (too generic to block on)
"""

import argparse
import csv
import os
import pickle
import re
import sys
import time
import unicodedata
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

LEGAL_SUFFIXES = {
    "inc", "llc", "ltd", "co", "corp", "corporation", "company", "pvt",
    "private", "limited", "group", "llp", "plc", "enterprises", "enterprise",
    "associates", "partners", "holdings", "industries", "international",
    "intl", "incorporated", "gmbh", "sa", "srl", "bv", "the", "and", "of",
    # French Additions
    "sarl", "sas", "sasu", "eurl", "sci", "societe", "responsabilite", "limitee"
}

TOKEN_RE = re.compile(r"[a-z0-9]+")

ADDRESS_STOPWORDS = {
    "street", "st", "ave", "avenue", "drive", "dr", "road", "rd", "blvd",
    "boulevard", "lane", "ln", "way", "court", "ct", "suite", "ste",
    "apartment", "apt", "unit", "po", "box", "floor", "building", "plaza",
    "trail", "circle", "cir", "place", "pl", "no", "north", "south", "east",
    "west", "n", "s", "e", "w",
    # French Additions
    "rue", "r", "boulevard", "bd", "avenue", "av", "allee", "chemin", "route", "place", "arrondissement"
}


def normalize_name(name):
    if not isinstance(name, str):
        return "", []
    s = unicodedata.normalize("NFKC", name).lower()
    tokens = TOKEN_RE.findall(s)
    sig_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES and len(t) >= 3]
    return " ".join(tokens), sig_tokens


def normalize_address(addr):
    if not isinstance(addr, str):
        return []
    s = unicodedata.normalize("NFKC", addr).lower()
    tokens = TOKEN_RE.findall(s)
    return [t for t in tokens if t not in ADDRESS_STOPWORDS and len(t) >= 3 and not t.isdigit()]


def normalize_country(c):
    if not isinstance(c, str):
        return "UNK"
    c = c.strip().lower()
    if c in ("us", "usa", "united states", "united states of america"):
        return "US"
    if c in ("india", "in"):
        return "IN"
    if c in ("france", "fr", "république française"):
        return "FR"
    return c.upper()


def load_source(path, is_source1=False):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country_norm"] = df["country"].apply(normalize_country)
    norm = df["business_name"].apply(normalize_name)
    df["name_norm"] = norm.apply(lambda x: x[0])
    df["name_tokens"] = norm.apply(lambda x: x[1])
    df["addr_tokens"] = df["business_address"].apply(normalize_address)
    return df


def build_index(df, token_col, max_posting):
    """country -> token -> list of row indices into df. Plain dicts only (picklable)."""
    index = {}
    counts = defaultdict(lambda: defaultdict(int))
    for i, tokens in enumerate(df[token_col]):
        country = df["country_norm"].iat[i]
        for t in set(tokens):
            counts[country][t] += 1

    for i, tokens in enumerate(df[token_col]):
        country = df["country_norm"].iat[i]
        for t in set(tokens):
            if counts[country][t] > max_posting:
                continue
            index.setdefault(country, {}).setdefault(t, []).append(i)
    return index


def build_index_cached(df, token_col, max_posting, cache_dir, cache_key):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}_{token_col}_{max_posting}.pkl")
    if os.path.exists(cache_path):
        print(f"  loading cached index: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    idx = build_index(df, token_col, max_posting)
    with open(cache_path, "wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  cached index to: {cache_path}")
    return idx


def candidates_for_row(tokens, country, index):
    seen = defaultdict(int)
    idx_for_country = index.get(country)
    if not idx_for_country:
        return seen
    for t in tokens:
        for i in idx_for_country.get(t, ()):
            seen[i] += 1
    return seen  # row_index -> token overlap count


def run(args):
    t0 = time.time()
    print("Loading Source 1...")
    s1 = load_source(args.source1)
    if args.sample:
        s1 = s1.sample(n=min(args.sample, len(s1)), random_state=42).reset_index(drop=True)
    print(f"  {len(s1)} Source 1 rows loaded")

    print("Loading Source 2...")
    s2 = load_source(args.source2)
    print(f"  {len(s2)} rows")

    print("Loading Source 3...")
    s3 = load_source(args.source3)
    print(f"  {len(s3)} rows")

    print("Building name index for Source 2...")
    idx2_name = build_index_cached(s2, "name_tokens", args.max_posting, args.cache_dir, "s2")
    print("Building address index for Source 2...")
    idx2_addr = build_index_cached(s2, "addr_tokens", args.max_posting, args.cache_dir, "s2")
    print("Building name index for Source 3...")
    idx3_name = build_index_cached(s3, "name_tokens", args.max_posting, args.cache_dir, "s3")
    print("Building address index for Source 3...")
    idx3_addr = build_index_cached(s3, "addr_tokens", args.max_posting, args.cache_dir, "s3")
    print(f"Indexing done in {time.time()-t0:.1f}s")

    cand_rows = []   # for candidate_pairs.tsv
    match_rows = []  # for matching_results.tsv

    n = len(s1)
    report_every = args.report_rows
    loop_start = time.time()

    for row_i in range(n):
        s1_id = s1["entity_id"].iat[row_i]
        s1_name_norm = s1["name_norm"].iat[row_i]
        s1_name_tokens = s1["name_tokens"].iat[row_i]
        s1_addr_tokens = s1["addr_tokens"].iat[row_i]
        s1_addr = s1["business_address"].iat[row_i]
        country = s1["country_norm"].iat[row_i]

        name_ov2 = candidates_for_row(s1_name_tokens, country, idx2_name)
        addr_ov2 = candidates_for_row(s1_addr_tokens, country, idx2_addr)
        name_ov3 = candidates_for_row(s1_name_tokens, country, idx3_name)
        addr_ov3 = candidates_for_row(s1_addr_tokens, country, idx3_addr)

        weight2 = defaultdict(int)
        for i, ov in name_ov2.items():
            weight2[i] += ov * 3
        for i, ov in addr_ov2.items():
            weight2[i] += ov
        weight3 = defaultdict(int)
        for i, ov in name_ov3.items():
            weight3[i] += ov * 3
        for i, ov in addr_ov3.items():
            weight3[i] += ov

        top2 = sorted(weight2.items(), key=lambda x: x[1], reverse=True)[: args.prefuzz_cap]
        top3 = sorted(weight3.items(), key=lambda x: x[1], reverse=True)[: args.prefuzz_cap]

        scored = []
        for i, w in top2:
            cand_id = s2["entity_id"].iat[i]
            name_score = fuzz.token_sort_ratio(s1_name_norm, s2["name_norm"].iat[i])
            addr_score = fuzz.token_sort_ratio(s1_addr, s2["business_address"].iat[i])
            combined = 0.75 * name_score + 0.25 * addr_score
            scored.append((cand_id, combined, w))
        for i, w in top3:
            cand_id = s3["entity_id"].iat[i]
            name_score = fuzz.token_sort_ratio(s1_name_norm, s3["name_norm"].iat[i])
            addr_score = fuzz.token_sort_ratio(s1_addr, s3["business_address"].iat[i])
            combined = 0.75 * name_score + 0.25 * addr_score
            scored.append((cand_id, combined, w))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: args.topk]

        cand_ids = [c[0] for c in top]
        cand_rows.append((s1_id, ",".join(cand_ids)))

        # --- DYNAMIC THRESHOLDING REPLACEMENT ---
        matched = []
        for c in top:
            score = c[1]
            if country == "US" and score >= 89.0:
                matched.append(c[0])
            elif country == "IN" and score >= 87.0:
                matched.append(c[0])
            elif country == "FR" and score >= 84.0:
                matched.append(c[0])
            elif score >= args.threshold: # Fallback threshold
                matched.append(c[0])
        # ----------------------------------------

        match_rows.append((s1_id, ",".join(matched)))

        if row_i % report_every == 0:
            elapsed = time.time() - t0
            loop_elapsed = time.time() - loop_start
            rate = loop_elapsed / (row_i + 1)
            full_eta_hr = rate * n / 3600
            print(f"  {row_i}/{n} ({100*row_i/n:.1f}%) elapsed {elapsed:.1f}s | "
                  f"{rate*1000:.1f}ms/row | full-run ETA: {full_eta_hr:.2f}h")

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    match_path = os.path.join(args.out_dir, "matching_results.tsv")

    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source1_entity_id", "candidate_entity_ids"])
        w.writerows(cand_rows)

    with open(match_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        w.writerows(match_rows)

    print(f"Wrote {cand_path}")
    print(f"Wrote {match_path}")
    print(f"Total time: {time.time()-t0:.1f}s")

    if args.eval and args.ground_truth:
        evaluate(args.ground_truth, s1, cand_rows, match_rows)


def evaluate(gt_path, s1, cand_rows, match_rows):
    print("\n--- Evaluation against ground truth ---")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt_map = {}
    for _, r in gt.iterrows():
        ids = r["matched_entity_ids"]
        gt_map[r["source1_entity_id"]] = set(ids.split(",")) if isinstance(ids, str) and ids else set()

    cand_map = {sid: set(c.split(",")) if c else set() for sid, c in cand_rows}
    match_map = {sid: set(m.split(",")) if m else set() for sid, m in match_rows}

    # Blocking recall: of true matches, how many appear in our candidate set?
    total_true = 0
    found_true = 0
    f_scores = []
    for s1_id in s1["entity_id"]:
        truth = gt_map.get(s1_id, set())
        cands = cand_map.get(s1_id, set())
        preds = match_map.get(s1_id, set())

        total_true += len(truth)
        found_true += len(truth & cands)

        if not truth:
            f_scores.append(1.0 if not preds else 0.0)
            continue
        tp = len(truth & preds)
        precision = tp / len(preds) if preds else 0.0
        recall = tp / len(truth)
        if precision == 0 and recall == 0:
            f = 0.0
        else:
            beta2 = 0.25
            denom = (beta2 * precision) + recall
            f = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
        f_scores.append(f)

    blocking_recall = found_true / total_true if total_true else 1.0
    avg_f = sum(f_scores) / len(f_scores) if f_scores else 0.0

    print(f"Blocking recall (true matches present in candidates): {blocking_recall:.4f}")
    print(f"Approx macro F_0.5 on this sample: {avg_f:.4f}")
    print("If blocking recall is low, matching quality is capped regardless of threshold.")
    print("If F_0.5 is much lower than blocking recall, raise --threshold (too many false merges).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source1", default="train_source1.tsv")
    p.add_argument("--source2", default="train_source2.tsv")
    p.add_argument("--source3", default="train_source3.tsv")
    p.add_argument("--ground-truth", default="train_ground_truth.tsv")
    p.add_argument("--sample", type=int, default=0, help="limit Source 1 rows for a quick test run")
    p.add_argument("--full", action="store_true", help="run on all Source 1 rows (ignores --sample)")
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--threshold", type=float, default=88.0)
    p.add_argument("--max-posting", type=int, default=800,
                    help="skip name/address tokens appearing in more than this many records (too generic)")
    p.add_argument("--prefuzz-cap", type=int, default=60,
                    help="max candidates per source kept (by token-overlap weight) before running fuzzy scoring")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--cache-dir", default="idx_cache",
                    help="folder to cache built indexes so retries with the same --max-posting skip rebuilding")
    p.add_argument("--report-rows", type=int, default=2000,
                    help="print progress every N rows (fixed count, not percentage)")
    p.add_argument("--eval", action="store_true", help="evaluate against ground truth after running")
    args = p.parse_args()

    if args.full:
        args.sample = 0

    run(args)
