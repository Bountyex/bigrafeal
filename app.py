import streamlit as st
import pandas as pd
import numpy as np
import random, math, heapq

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(page_title="⚡ 7 Number Lowest Result (FAST)", layout="wide")
st.title("⚡ 7 Number Lowest Result — FAST ENGINE")

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.stop()

# =========================================================
# READ + CLEAN
# =========================================================
df = pd.read_excel(uploaded_file)
df.columns = df.columns.astype(str).str.strip()

numbers_col = "Selected Number"
if numbers_col not in df.columns:
    st.error(f"❌ Column '{numbers_col}' not found")
    st.write("Found:", df.columns.tolist())
    st.stop()

# =========================================================
# PARSE TICKETS
# =========================================================
def parse_ticket(x):
    try:
        t = tuple(sorted(map(int, str(x).split(","))))
        if len(t) == 7 and all(1 <= n <= 37 for n in t):
            return t
    except:
        pass
    return None

tickets = df[numbers_col].dropna().apply(parse_ticket).dropna().tolist()
n = len(tickets)
if n == 0:
    st.error("No valid tickets")
    st.stop()

st.success(f"Loaded {n} tickets")

# =========================================================
# CACHED INDICATOR MATRIX
# =========================================================
@st.cache_data(show_spinner=False)
def build_indicator(tickets):
    M = np.zeros((len(tickets), 37), dtype=np.uint8)
    for i, t in enumerate(tickets):
        M[i, np.array(t) - 1] = 1
    return M

ind = build_indicator(tickets)

# =========================================================
# PAYOUT VECTOR
# =========================================================
payout = np.array([0, 0, 0, 15, 1000, 4000, 10000, 100000], dtype=np.int64)

# =========================================================
# FAST SCORER (DOT PRODUCT)
# =========================================================
def score_candidate(mask):
    counts = ind @ mask
    return payout[counts].sum()

# =========================================================
# FREQUENCY HEURISTIC
# =========================================================
freq = ind.sum(axis=0)
low_nums = np.argsort(freq)[:14]

# =========================================================
# NEIGHBOR MOVE (NO ALLOCATIONS)
# =========================================================
def neighbor(mask):
    new = mask.copy()
    ones = np.flatnonzero(new)
    zeros = np.flatnonzero(new == 0)
    new[random.choice(ones)] = 0
    new[random.choice(zeros)] = 1
    return new

# =========================================================
# SIMULATED ANNEALING (FAST)
# =========================================================
def fast_sa(mask, iters=1500, t0=5.0):
    best = curr = mask
    best_s = curr_s = score_candidate(curr)

    for i in range(iters):
        T = t0 * (1 - i / iters)
        cand = neighbor(curr)
        s = score_candidate(cand)
        d = s - curr_s

        if d < 0 or random.random() < math.exp(-d / (T + 1e-9)):
            curr, curr_s = cand, s
            if s < best_s:
                best, best_s = cand, s
    return best, best_s

# =========================================================
# INITIAL POPULATION
# =========================================================
def make_mask(nums):
    m = np.zeros(37, dtype=np.uint8)
    m[nums] = 1
    return m

candidates = []

for _ in range(30):
    candidates.append(make_mask(random.sample(list(low_nums), 7)))

for _ in range(40):
    candidates.append(make_mask(random.sample(range(37), 7)))

candidates.append(make_mask(low_nums[:7]))

# =========================================================
# RUN
# =========================================================
if st.button("🚀 Run Fast Optimization"):
    st.info("Optimizing…")

    bar = st.progress(0)
    heap = []

    for i, c in enumerate(candidates):
        b, s = fast_sa(c)
        heapq.heappush(heap, (s, b))
        heap = heapq.nsmallest(20, heap)
        bar.progress((i + 1) / len(candidates))

    # =====================================================
    # FINAL LOCAL REFINEMENT
    # =====================================================
    results = []

    for s, m in heap[:10]:
        best_m, best_s = m, s
        for _ in range(2000):
            cand = neighbor(best_m)
            sc = score_candidate(cand)
            if sc < best_s:
                best_m, best_s = cand, sc
        results.append((best_s, best_m))

    results.sort()

    # =====================================================
    # DISPLAY
    # =====================================================
    st.subheader("🏆 Top 10 Lowest-Payout Results")

    for s, m in results:
        combo = tuple(np.where(m == 1)[0] + 1)
        st.write(f"**{combo} → ₹{s:,}**")

    best_s, best_m = results[0]
    best_combo = tuple(np.where(best_m == 1)[0] + 1)

    counts = ind @ best_m
    uniq, cnt = np.unique(counts, return_counts=True)

    st.subheader("📊 Breakdown")
    st.write(dict(zip(uniq.astype(int), cnt.astype(int))))
    st.success(f"🔥 Best: {best_combo} → ₹{best_s:,}")
