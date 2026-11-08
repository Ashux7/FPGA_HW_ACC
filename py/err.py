"""
exhaustive error analysis of 8-bit unsigned multipliers and adders.

Models:
  exact_mult      : the normal, correct a*b
  trunc_operand   : zero the low K bits of BOTH inputs, then multiply
  trunc_columns   : build the 8x8 partial-product grid, delete every cell
                    whose column index (i + j) is below K, then add the rest
  exact_add       : the normal, correct a+b
  LOA_Add         : lower-part OR adder (adds upper WIDTH-K bits, ORs lower K bits)

Metrics (computed over ALL 65,536 input pairs):
  ER   error rate            = fraction of inputs with a wrong answer
  MED  mean error distance   = average |approx - exact|
  MRED mean relative ED      = average |approx - exact| / exact  (exact != 0)
  WCE  worst-case error      = max |approx - exact|

"""

import csv

WIDTH = 8
N = 1 << WIDTH  # 256 values per operand


def exact_mult(a, b):
    return a * b


def exact_add(a, b):
    return a + b


def LOA_Add(a, b, k):
    """
    Lower-Part OR Adder (LOA):
    - Adds the upper (WIDTH - K) bits exactly
    - Performs bitwise OR on the lower K bits
    """
    mask = (1 << k) - 1
    lower = (a & mask) | (b & mask)
    upper = (a >> k) + (b >> k)
    return (upper << k) | lower


def trunc_operand(a, b, k):
    mask = ~((1 << k) - 1)          # e.g. k=2 -> ...11111100
    return (a & mask) * (b & mask)


def trunc_columns(a, b, k):
    total = 0
    for i in range(WIDTH):              # bit i of a
        for j in range(WIDTH):          # bit j of b
            if i + j >= k:              # keep only columns >= k
                total += (((a >> i) & 1) & ((b >> j) & 1)) << (i + j)
    return total


def evaluate(approx_fn, k, exact_fn=exact_mult):
    errors = 0
    sum_ed = 0
    sum_red = 0.0
    red_count = 0
    wce = 0
    for a in range(N):
        for b in range(N):
            ex = exact_fn(a, b)
            ap = approx_fn(a, b, k)
            ed = abs(ap - ex)
            if ed != 0:
                errors += 1
            sum_ed += ed
            wce = max(wce, ed)
            if ex != 0:
                sum_red += ed / ex
                red_count += 1
    total = N * N
    return {
        "ER": errors / total,
        "MED": sum_ed / total,
        "MRED": sum_red / red_count,
        "WCE": wce,
    }


def main():
    # 1. Multiplier analysis
    mult_designs = [("trunc_operand", trunc_operand), ("trunc_columns", trunc_columns)]
    mult_rows = []
    print(f"{'=== 8-bit Multipliers ===':^62}")
    print(f"{'design':<15}{'K':>3}{'ER':>10}{'MED':>12}{'MRED':>12}{'WCE':>10}")
    for name, fn in mult_designs:
        for k in range(0, 7):
            m = evaluate(fn, k, exact_fn=exact_mult)
            mult_rows.append({"design": name, "K": k, **m})
            print(f"{name:<15}{k:>3}{m['ER']:>10.4f}{m['MED']:>12.2f}"
                  f"{m['MRED']:>12.5f}{m['WCE']:>10d}")

    with open("error_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["design", "K", "ER", "MED", "MRED", "WCE"])
        w.writeheader()
        w.writerows(mult_rows)
    print("\nSaved error_table.csv")

    # 2. LOA Adder analysis
    loa_designs = [("LOA_Add", LOA_Add)]
    loa_rows = []
    print(f"\n{'=== 8-bit LOA Adder ===':^62}")
    print(f"{'design':<15}{'K':>3}{'ER':>10}{'MED':>12}{'MRED':>12}{'WCE':>10}")
    for name, fn in loa_designs:
        for k in range(0, 8):
            m = evaluate(fn, k, exact_fn=exact_add)
            loa_rows.append({"design": name, "K": k, **m})
            print(f"{name:<15}{k:>3}{m['ER']:>10.4f}{m['MED']:>12.2f}"
                  f"{m['MRED']:>12.5f}{m['WCE']:>10d}")

    with open("loa_error_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["design", "K", "ER", "MED", "MRED", "WCE"])
        w.writeheader()
        w.writerows(loa_rows)
    print("\nSaved loa_error_table.csv")

    # 3. Plots (skipped silently if matplotlib is not installed)
    try:
        import matplotlib.pyplot as plt

        # Multiplier plot
        plt.figure()
        for name, _ in mult_designs:
            ks = [r["K"] for r in mult_rows if r["design"] == name]
            mred = [r["MRED"] for r in mult_rows if r["design"] == name]
            plt.plot(ks, mred, marker="o", label=name)
        plt.xlabel("K (approximated bits / columns)")
        plt.ylabel("MRED")
        plt.yscale("log")
        plt.title("8-bit Multiplier Error vs K")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig("mred_vs_k.png", dpi=150)
        plt.close()
        print("Saved mred_vs_k.png")

        # LOA Adder plot
        plt.figure()
        for name, _ in loa_designs:
            ks = [r["K"] for r in loa_rows if r["design"] == name]
            mred = [r["MRED"] for r in loa_rows if r["design"] == name]
            plt.plot(ks, mred, marker="s", color="crimson", label=name)
        plt.xlabel("K (approximate bits in lower part)")
        plt.ylabel("MRED")
        plt.yscale("log")
        plt.title("8-bit LOA Adder Error vs K")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig("loa_mred_vs_k.png", dpi=150)
        plt.close()
        print("Saved _loa_mred_vs_k.png")

    except ImportError:
        print("matplotlib not installed; skipped plot (pip install matplotlib)")


if __name__ == "__main__":
    main()
