import pandas as pd
import numpy as np
import random, math, heapq, time
import streamlit as st

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(page_title="7 Number Lowest Result", layout="wide")
st.title("🎯 7 Number Lowest Result")

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.stop()

# =========================================================
# READ EXCEL
# =========================================================
df = pd.read_excel(uploaded_file)

# =========================================================
# CLEAN & VALIDATE COLUMN NAME
# =========================================================
df.columns = (
    df.columns
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

if "Selected Numbers" not in df.columns:
    st.error("❌ Column 'Selected Numbers' not found.")
    st.write("📌 Found columns:", df.columns.tolist())
    st.stop()

# =========================================================
# PARSE TICKETS
# =========================================================
def parse_ticket(x):
    try:
        nums = sorted(map(int, str(x).split(",")))
        if len(nums) != 7:
            return None
        if any(n < 1 or n > 37 for n in nums):
            return None
        return tuple(nums)
    except:
        return None

tickets = df["Selected Numbers"].dropna().apply(parse_ticket).dropna().tolist()
n_tickets = len(tickets)

if n_tickets == 0:
    st.error("❌ No valid tickets found.")
    st.stop()

st.success(f"✅ Loaded {n_tickets} valid tickets")

# =========================================================
# BUILD INDICATOR MATRIX
# =========================================================
cols = 37
ind = np.zeros((n_tickets, cols), dtype=np.uint8)
for i, t in enumerate(tickets):
    ind[i, np.array(t) - 1] = 1

# =========================================================
# PAYOUT TABLE
# =========================================================
payout_lookup = np.zeros(8, dtype=np.int64)
payout_lookup[3] = 15
payout_lookup[4] = 1000
payout_lookup[5] = 4000
payout_lookup[6] = 10000
payout_lookup[7] = 100000

def total_payout_for_candidate(candidate):
    idx = np.array(candidate) - 1
    counts = ind[:, idx].sum(axis=1)
    return int(np.sum(payout_lookup[counts]))

# =========================================================
# FREQUENCY HEURISTIC
# =========================================================
freq = ind.sum(axis=0)
freq_sorted = np.argsort(freq)
low_freq_nums = [int(i + 1) for i in freq_sorted[:14]]

# =========================================================
# NEIGHBOR MOVE
# =========================================================
def neighbor_swap(candidate):
    cand = list(candidate)
    out_nums = [n for n in range(1, cols + 1) if n not in cand]
    remove = random.choice(cand)
    add = random.choice(out_nums)
    cand[cand.index(remove)] = add
    return tuple(sorted(cand))

# =========================================================
# SIMULATED ANNEALING
# =========================================================
def hillclimb_sa(initial, max_iters=2500, start_temp=8.0, end_temp=0.001):
    best = current = initial
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

# =========================================================
# INITIAL CANDIDATES
# =========================================================
candidates = [tuple(sorted(random.sample(low_freq_nums, 7))) for _ in range(40)]
candidates += [tuple(sorted(random.sample(range(1, cols + 1), 7))) for _ in range(60)]
candidates.append(tuple(int(i + 1) for i in freq_sorted[:7]))
candidates = list(dict.fromkeys(candidates))

# =========================================================
# RUN OPTIMIZATION
# =========================================================
st.info("⏳ Running optimization… please wait")

top_candidates = []
start = time.time()

for c in candidates:
    best_c, score = hillclimb_sa(c)
    heapq.heappush(top_candidates, (score, best_c))
    if len(top_candidates) > 30:
        top_candidates = heapq.nsmallest(30, top_candidates)

# =========================================================
# FINAL LOCAL SEARCH
# =========================================================
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

# =========================================================
# DISPLAY RESULTS
# =========================================================
st.subheader("✅ 10 Lowest-Payout Combinations")
for score, combo in final_results:
    st.write(f"**{combo} → ₹{score:,}**")

# =========================================================
# BREAKDOWN
# =========================================================
best_score, best_combo = final_results[0]
st.subheader("📊 Breakdown of Best Combination")

best_counts = ind[:, np.array(best_combo) - 1].sum(axis=1)
unique, counts = np.unique(best_counts, return_counts=True)
breakdown = {int(k): int(v) for k, v in zip(unique, counts)}

st.write(breakdown)

st.success(f"🏆 Best Combination: {best_combo} → ₹{best_score:,}")
