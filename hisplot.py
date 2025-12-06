import matplotlib.pyplot as plt

# =========================
# Data
# =========================

# Greedy (equivalent to Temp=0, top-p=0)
greedy_bins = [324, 0, 0, 0, 0, 0, 0, 76]  # 0–25, 25–50, ..., 175–200
# Probabilistic Temp=1.0, top-p=1.0
prob_bins = [236, 49, 29, 20, 15, 13, 17, 21]

bin_labels = [
    "0–25", "25–50", "50–75", "75–100",
    "100–125", "125–150", "150–175", "175–200"
]

# =========================
# Plot histogram (a) Greedy
# =========================
plt.figure(figsize=(8, 4))
plt.bar(bin_labels, greedy_bins)
plt.xticks(rotation=45)
plt.ylabel("Number of Questions")
# plt.title("(a) Greedy Decoding")
plt.tight_layout()
plt.savefig("hist_greedy.png", dpi=300)
plt.close()

# =========================
# Plot histogram (b) Probabilistic
# =========================
plt.figure(figsize=(8, 4))
plt.bar(bin_labels, prob_bins)
plt.xticks(rotation=45)
plt.ylabel("Number of Questions")
# plt.title("(b) Temp=1.0, top-p=1.0")
plt.tight_layout()
plt.savefig("hist_prob.png", dpi=300)
plt.close()

print("Saved hist_greedy.png and hist_prob.png")
