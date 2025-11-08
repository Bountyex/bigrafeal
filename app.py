import pandas as pd
import numpy as np
import random, math, heapq, time
import streamlit as st

st.title("🎯 7 number Lowest Result")

uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)
tickets = df["Selected Numbers"].apply(lambda x: tuple(sorted(map(int, str(x).split(","))))).tolist()
n_tickets = len(tickets)

cols = 37
ind = np.zeros((n_tickets, cols), dtype=np.uint8)
for i, t in enumerate(tickets):
    ind[i, np.array(t) - 1] = 1

payout_lookup = np.zeros(8, dtype=np.int64)
payout_lookup[3] = 15
payout_lookup[4] = 1000
payout_lookup[5] = 4000
payout_lookup[6] = 10000
payout_lookup[7] = 100000

def total_payout_for_candidate(candidate_tuple):
    cols_idx = np.fromiter((c-1 for c in candidate_tuple), dtype=np.int8)
    counts = ind[:, cols_idx].sum(axis=1)
    return int(np.sum(payout_lookup[counts]))

freq = ind.sum(axis=0)
freq_sorted = np.argsort(freq)
low_freq_nums = [int(i + 1) for i in freq_sorted[:14]]

def neighbor_swap(candidate):
    cand = list(candidate)
    out_nums = [n for n in range(1, cols + 1) if n not in cand]
    remove = random.choice(cand)
    add = random.choice(out_nums)
    cand[cand.index(remove)] = add
    return tuple(sorted(cand))

def hillclimb_sa(initial_candidate, max_iters=2500, start_temp=8.0, end_temp=0.001):
    best = current = initial_candidate
    best_score = current_score = total_payout_for_candidate(current)
    for it in range(max_iters):
        T = start_temp * ((end_temp / start_temp) ** (it / max_iters))
        cand = neighbor_swap(current)
        sc = total_payout_for_candidate(cand)
        delta = sc - current_score
        if delta < 0 or random.random() < math.exp(-delta / (T + 1e-9)):
            current, current_score = cand, sc
            if sc < best_score:
                best, best_score = cand, sc
    return best, best_score

candidates_to_try = [tuple(sorted(random.sample(low_freq_nums, 7))) for _ in range(40)]
candidates_to_try += [tuple(sorted(random.sample(range(1, cols + 1), 7))) for _ in range(60)]
candidates_to_try.append(tuple(int(i + 1) for i in freq_sorted[:7]))
candidates_to_try = list(dict.fromkeys(candidates_to_try))

st.write("⏳ Running optimization... Please wait.")

top_candidates = []
start = time.time()

for c in candidates_to_try:
    cand_best, cand_score = hillclimb_sa(c)
    heapq.heappush(top_candidates, (cand_score, cand_best))
    if len(top_candidates) > 30:
        top_candidates = heapq.nsmallest(30, top_candidates)

final_results = []
for score, cand in top_candidates[:10]:
    best_local, best_score = cand, score
    for _ in range(10000):
        cand2 = neighbor_swap(best_local)
        sc = total_payout_for_candidate(cand2)
        if sc < best_score:
            best_local, best_score = cand2, sc
    final_results.append((best_score, best_local))

final_results = sorted(final_results)[:10]

st.subheader("✅ 10 Lowest-Payout Combinations")
for score, combo in final_results:
    st.write(f"{combo} → **₹{score:,}**")

best_score, best_combo = final_results[0]
st.subheader("📊 Detailed Breakdown of Best Combination")
best_counts = ind[:, np.array(best_combo) - 1].sum(axis=1)
unique, counts = np.unique(best_counts, return_counts=True)
breakdown = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
st.write("Match Breakdown:", breakdown)
