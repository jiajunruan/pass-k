import json
import numpy as np
import matplotlib.pyplot as plt
import os

# 要画的配置
subfolder_config_names = [
    "NPO", "SimNPO", "GradDiff", "RMU"
]

# 固定的 k 值
k_values = [2 ** i for i in range(8)]  # [1,2,4,8,16,32,64,128]

BASE_DIR = "/users/2/jruan/pass-k/saves/eval"
OUT_DIR = "fixplotting"   # <<< 新增：统一输出目录
os.makedirs(OUT_DIR, exist_ok=True)


def load_rougeL(path):
    if not os.path.isfile(path):
        print(f"Warning: file not found: {path}")
        return None

    try:
        with open(path, "r") as f:
            jd = json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

    mapping = {}
    for k, v in jd.items():
        try:
            mapping[int(k)] = float(v)
        except Exception:
            pass

    return [mapping.get(k, np.nan) for k in k_values]


for subfolder in subfolder_config_names:
    plt.figure(figsize=(10, 6))

    paths = {
        "Original": f"{BASE_DIR}/{subfolder}/forget/temperature=1.0top_p=1.0/rougeL_summary.json",
        "Fix": f"{BASE_DIR}/{subfolder}-2_fix/forget/temperature=1.0top_p=1.0/rougeL_summary.json",
    }

    styles = {
        "Original": dict(color="blue", marker="o", linestyle="-"),
        "Fix": dict(color="red", marker="s", linestyle="--"),
    }

    plotted_any = False

    for name, path in paths.items():
        y = load_rougeL(path)
        if y is None:
            continue

        plt.plot(
            k_values,
            y,
            label=name,
            linewidth=2,
            markersize=6,
            **styles[name],
        )
        plotted_any = True

    if not plotted_any:
        print(f"Skip plotting {subfolder}: no valid data.")
        plt.close()
        continue

    plt.xscale("log", base=2)
    plt.xticks(k_values, [str(k) for k in k_values], fontsize=19)
    plt.yticks(np.linspace(0, 1, 11), fontsize=19)

    plt.ylim(0, 1)
    plt.xlabel("Number of Generations $k$", fontsize=19)
    plt.ylabel("$\widehat{leak@k}$-ES", fontsize=19)

    plt.title(subfolder, fontsize=20)
    plt.legend(loc="best", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"rougeL_{subfolder}.png")
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved plot to {out_png}")
