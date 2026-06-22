# =========================================================
# LATENT CHAIN-OF-THOUGHT — TOKEN EFFICIENCY DASHBOARD
# Streamlit App
#
# Menggabungkan dua skrip analisis:
#   1. GSM8K-style dataset (.parquet)
#   2. SVAMP dataset (.csv, dengan kolom Equation)
#
# Membandingkan tiga mode inferensi:
#   - No CoT     : jawaban langsung
#   - Full CoT   : reasoning eksplisit + jawaban akhir
#   - Latent CoT : reasoning implisit, jawaban didistilasi
#
# =========================================================

import os
import re
from fractions import Fraction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# =========================================================
# 0. KONFIGURASI PATH DATASET
# =========================================================
# Path dibuat RELATIF terhadap lokasi app.py (bukan hardcode absolut),
# sehingga otomatis mencari file di folder yang sama dengan app.py —
# baik dijalankan lokal maupun setelah di-deploy dari repo GitHub
# (mis. Streamlit Community Cloud), asalkan dataset diletakkan di
# folder yang sama (atau subfolder) dengan app.py di repo tersebut.
#
# Jika dataset Anda ada di subfolder repo (misal "data/"), cukup ubah
# nama file di bawah ini, contoh:
#   GSM8K_FILENAME = "data/train-00000-of-00001.parquet"

APP_DIR = os.path.dirname(os.path.abspath(__file__))

GSM8K_FILENAME = "train-00000-of-00001.parquet"
SVAMP_FILENAME = "SVAMP.csv"

GSM8K_PATH = os.path.join(APP_DIR, GSM8K_FILENAME)
SVAMP_PATH = os.path.join(APP_DIR, SVAMP_FILENAME)

st.set_page_config(
    page_title="Latent CoT — Token Efficiency Dashboard",
    page_icon="🧠",
    layout="wide",
)

# =========================================================
# 1. DARK THEME HELPERS (matplotlib)
# =========================================================

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#30363d"
WHITE    = "white"
MUTED    = "#8b949e"
COLORS   = ["#5b8dee", "#e05c5c", "#2dd68b"]
MODE_LABELS = ["No CoT", "Full CoT\n(Standard)", "Latent CoT\n(Proposed)"]
MODE_LABELS_SHORT = ["No CoT", "Full CoT", "Latent CoT"]
MODES = ["no_cot", "cot", "latent_cot"]


def dark_ax(ax, title="", xlabel="", ylabel="", grid_axis="y"):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=WHITE, fontsize=12, pad=10, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=WHITE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.grid(axis=grid_axis, color=GRID_COL, linewidth=0.5, zorder=0)


def new_fig(figsize):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    return fig


# =========================================================
# 2. TOKENIZER (regex sederhana, tanpa dependency eksternal)
# =========================================================

_TOKEN_PATTERN = re.compile(r"[\w']+|[.,!?;:()\[\]{}\-$%]")


def count_tokens(text) -> int:
    if pd.isna(text) or text is None:
        return 0
    return len(_TOKEN_PATTERN.findall(str(text)))


# =========================================================
# 3. ANSWER EXTRACTION — MULTI-PASS (dipakai semua dataset)
# =========================================================

EXTRACTION_PATTERNS = [
    r"####\s*([\-\d,\.]+)",
    r"(?:the\s+)?(?:final\s+)?answer\s+(?:is|:)\s*([\-\d,\.]+)",
    r"(?:therefore|thus|so)[,\s]+(?:\w+\s+){0,5}([\-\d,\.]+)",
    r"(?:total|result|value|sum|difference|product)\s*(?:is|=|:)\s*([\-\d,\.]+)",
    r"=\s*([\-\d,\.]+)\s*(?:$|\.|\n)",
    r"\$\s*([\d,\.]+)",
    r"([\-\d,\.]+)\s*(?:dollar|cent|hour|day|week|year|kg|km|meter|mile|gallon|"
    r"pound|foot|feet|inch|yard|percent|%)s?\s*\.?\s*$",
    r"([\-\d,\.]+)\s*$",
]


def normalize_number(raw: str):
    clean = raw.strip().replace(",", "").rstrip(".")
    try:
        return float(clean)
    except ValueError:
        return None


def try_eval_equation(eq_text: str):
    """Evaluasi ekspresi matematika sederhana (hanya angka + operator dasar)."""
    clean = re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", "", str(eq_text)).strip()
    if not clean:
        return None
    try:
        result = eval(clean, {"__builtins__": {}})
        return float(result)
    except Exception:
        return None


def extract_final_answer(answer_text: str, equation_text: str = "") -> str:
    """
    Multi-pass extraction.
    Pass-1 : 8 pola regex bertingkat
    Pass-2 : Evaluasi Equation jika tersedia (equation-aware, hanya jika kolom ada)
    Pass-3 : Angka manapun di teks, ambil yang terakhir
    Pass-4 : Baris terakhir (maks 10 kata)
    Pass-5 : Truncate (maks 10 kata)
    """
    text = str(answer_text).strip()
    eq = str(equation_text).strip() if equation_text else ""

    for pat in EXTRACTION_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = normalize_number(m.group(1).strip())
            if val is not None:
                return str(int(val)) if val == int(val) else str(round(val, 4))

    if eq:
        eq_val = try_eval_equation(eq)
        if eq_val is not None:
            return str(int(eq_val)) if eq_val == int(eq_val) else str(round(eq_val, 4))

    all_nums = re.findall(r"[\-]?\d[\d,\.]*", text)
    if all_nums:
        val = normalize_number(all_nums[-1])
        if val is not None:
            return str(int(val)) if val == int(val) else str(round(val, 4))

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return " ".join(lines[-1].split()[:10])

    return " ".join(text.split()[:10])


def extract_number(text: str):
    clean = str(text).replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", clean)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


# =========================================================
# 4. ANSWER DISTILLATION (Latent CoT core)
# =========================================================

ANSWER_KEYWORDS = {
    "total", "answer", "final", "result", "therefore", "thus",
    "cost", "earn", "spend", "left", "remain", "need", "have",
    "make", "give", "pay", "buy", "sell", "receive", "took",
    "equals", "is", "are", "was", "were",
}


def token_score(word: str) -> float:
    w = word.strip().lower()
    if re.search(r"\d", w):
        return 3.0
    if w in ANSWER_KEYWORDS:
        return 2.0
    if re.match(r"[.,!?;:]", w):
        return 0.05
    if len(w) <= 2:
        return 0.2
    return 1.0


def distill_latent_answer(answer_text: str, equation_text: str = "",
                           max_words: int = 15) -> str:
    """
    Distilasi jawaban untuk output Latent CoT. Jika equation_text kosong
    (mis. dataset GSM8K tanpa kolom Equation), perilaku otomatis fallback
    ke ekstraksi multi-pass biasa (tanpa override equation-aware).
    """
    eq_val = try_eval_equation(equation_text) if equation_text else None
    core = extract_final_answer(answer_text, equation_text)

    if eq_val is not None:
        core_num = extract_number(core)
        if core_num is None or abs(core_num - eq_val) / (abs(eq_val) + 1e-9) > 0.01:
            core = str(int(eq_val)) if eq_val == int(eq_val) else str(round(eq_val, 4))

    if re.fullmatch(r"-?\d+(\.\d+)?", core.strip()):
        return core.strip()

    words = core.split()
    if len(words) <= max_words:
        return core

    scored = [(i, w, token_score(w)) for i, w in enumerate(words)]
    top = sorted(scored, key=lambda x: x[2], reverse=True)[:max_words]
    top = sorted(top, key=lambda x: x[0])
    return " ".join(w for _, w, _ in top)


# =========================================================
# 5. SIMULASI OUTPUT PER MODE (generik untuk semua dataset)
# =========================================================

def simulate_output(question: str, answer: str, reasoning: str, mode: str,
                     equation: str = "") -> str:
    if mode == "no_cot":
        return f"Q: {question}\nA: {answer}"
    elif mode == "cot":
        return f"Q: {question}\nReasoning: {reasoning}\nFinal Answer: {answer}"
    elif mode == "latent_cot":
        distilled = distill_latent_answer(answer, equation_text=equation, max_words=20)
        return f"Q: {question}\nA: {distilled}"
    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")


# =========================================================
# 6. SOFT-MATCH ACCURACY
# =========================================================

def robust_normalize(text: str) -> str:
    raw = extract_final_answer(str(text))
    frac_match = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", raw.strip())
    if frac_match:
        try:
            val = float(Fraction(int(frac_match.group(1)), int(frac_match.group(2))))
            return str(int(val)) if float(val).is_integer() else f"{val:.6f}".rstrip("0")
        except Exception:
            pass
    num_match = re.search(r"-?[\d,\.]+", raw)
    if num_match:
        try:
            val = float(num_match.group().replace(",", ""))
            return str(int(val)) if val.is_integer() else f"{val:.6f}".rstrip("0")
        except ValueError:
            pass
    return re.sub(r"\s+", " ", raw).strip().lower()


def soft_match(pred: str, gt: str) -> bool:
    if pred == gt:
        return True
    try:
        p_val = float(pred)
        g_val = float(gt)
        if abs(p_val - g_val) <= 1e-6:
            return True
        if p_val == int(p_val) and g_val == int(g_val):
            return abs(p_val - g_val) <= 1
    except ValueError:
        pass
    return False


def calculate_accuracy_for_mode(predicted_outputs: pd.Series, ground_truth: pd.Series) -> dict:
    pred_norm = predicted_outputs.apply(robust_normalize)
    gt_norm = ground_truth.apply(robust_normalize)
    correct_mask = pd.Series(
        [soft_match(p, g) for p, g in zip(pred_norm, gt_norm)],
        index=predicted_outputs.index,
    )
    return {
        "accuracy": correct_mask.mean() * 100,
        "correct": int(correct_mask.sum()),
        "total": len(correct_mask),
        "series": correct_mask,
    }


# =========================================================
# 7. LOADER DATASET (deteksi kolom otomatis untuk format generik,
#    skema tetap untuk SVAMP)
# =========================================================

POSSIBLE_Q_COLS = ["question", "input", "prompt", "problem"]
POSSIBLE_A_COLS = ["answer", "output", "response", "solution"]
POSSIBLE_R_COLS = ["reasoning", "rationale", "explanation", "steps"]


def find_column(possible_list, df_cols):
    for col in possible_list:
        if col in df_cols:
            return col
    return None


@st.cache_data(show_spinner=False)
def load_gsm8k(path: str):
    if not os.path.exists(path):
        return None, f"File tidak ditemukan: {path}"
    df = pd.read_parquet(path)
    q_col = find_column(POSSIBLE_Q_COLS, df.columns)
    a_col = find_column(POSSIBLE_A_COLS, df.columns)
    r_col = find_column(POSSIBLE_R_COLS, df.columns)
    if q_col is None or a_col is None:
        return None, "Kolom question/answer tidak ditemukan di dataset."

    out = pd.DataFrame()
    out["problem"] = df[q_col].astype(str)
    out["answer_str"] = df[a_col].astype(str)
    out["reasoning_str"] = df[r_col].astype(str) if r_col else ""
    out["equation_str"] = ""  # GSM8K tidak punya kolom Equation
    out["Type"] = "GSM8K"
    return out, None


@st.cache_data(show_spinner=False)
def load_svamp(path: str):
    if not os.path.exists(path):
        return None, f"File tidak ditemukan: {path}"
    df = pd.read_csv(path)
    required = {"Body", "Question", "Answer", "Equation"}
    missing = required - set(df.columns)
    if missing:
        return None, f"Kolom berikut tidak ditemukan di SVAMP.csv: {missing}"

    out = pd.DataFrame()
    out["problem"] = df["Body"].astype(str).str.strip() + " " + df["Question"].astype(str).str.strip()
    out["answer_str"] = df["Answer"].astype(str)
    out["reasoning_str"] = df["Equation"].astype(str)  # dipakai sbg "reasoning" utk mode CoT
    out["equation_str"] = df["Equation"].astype(str)
    out["Type"] = df["Type"] if "Type" in df.columns else "SVAMP"
    return out, None


# =========================================================
# 8. PIPELINE ANALISIS (dipakai untuk dataset manapun, setelah dinormalisasi)
# =========================================================

@st.cache_data(show_spinner=False)
def run_analysis(df: pd.DataFrame, max_distill_words: int):
    df = df.copy()

    df["question_tokens"] = df["problem"].apply(count_tokens)
    df["answer_tokens"] = df["answer_str"].apply(count_tokens)
    df["reasoning_tokens"] = df["reasoning_str"].apply(count_tokens)

    results = {}
    for mode in MODES:
        outputs = df.apply(
            lambda row: simulate_output(
                row["problem"], row["answer_str"], row["reasoning_str"], mode,
                equation=row["equation_str"],
            ),
            axis=1,
        )
        token_series = outputs.apply(count_tokens)
        results[mode] = {
            "mean": token_series.mean(),
            "median": token_series.median(),
            "max": token_series.max(),
            "min": token_series.min(),
            "total": token_series.sum(),
            "series": token_series,
            "outputs": outputs,
        }
        df[f"tokens_{mode}"] = token_series

    # Akurasi soft-match (No CoT & Full CoT akan ~100% karena outputnya memuat answer asli)
    accuracy_results = {}
    for mode in MODES:
        accuracy_results[mode] = calculate_accuracy_for_mode(results[mode]["outputs"], df["answer_str"])
        df[f"correct_{mode}"] = accuracy_results[mode]["series"]

    return df, results, accuracy_results


def compute_efficiency(results: dict) -> dict:
    no_cot_mean = results["no_cot"]["mean"]
    cot_mean = results["cot"]["mean"]
    latent_mean = results["latent_cot"]["mean"]

    overhead_cot = ((cot_mean - no_cot_mean) / no_cot_mean) * 100 if no_cot_mean else 0
    reduction_cot_lat = ((cot_mean - latent_mean) / cot_mean) * 100 if cot_mean else 0
    reduction_nocot_lat = (
        ((no_cot_mean - latent_mean) / no_cot_mean) * 100
        if no_cot_mean and latent_mean < no_cot_mean else 0
    )
    total_cot = results["cot"]["total"]
    total_latent = results["latent_cot"]["total"]
    total_reduction = ((total_cot - total_latent) / total_cot) * 100 if total_cot else 0

    return {
        "no_cot_mean": no_cot_mean,
        "cot_mean": cot_mean,
        "latent_mean": latent_mean,
        "overhead_cot": overhead_cot,
        "reduction_cot_lat": reduction_cot_lat,
        "reduction_nocot_lat": reduction_nocot_lat,
        "total_cot": total_cot,
        "total_latent": total_latent,
        "total_reduction": total_reduction,
    }


# =========================================================
# 9. UI — SIDEBAR
# =========================================================

st.title("🧠 Latent Chain-of-Thought — Token Efficiency Dashboard")
st.caption(
    "Membandingkan efisiensi token & akurasi antara **No CoT**, **Full CoT**, dan "
    "**Latent CoT** pada dataset math word problem."
)

with st.sidebar:
    st.header("⚙️ Pengaturan")

    dataset_choice = st.radio(
        "Pilih dataset",
        ["GSM8K-style (.parquet)", "SVAMP (.csv)"],
        help="File dicari otomatis di folder yang sama dengan app.py "
             "(termasuk saat di-deploy dari repo GitHub). Ubah GSM8K_FILENAME / "
             "SVAMP_FILENAME di bagian atas skrip jika nama/lokasi file berbeda.",
    )

    st.code(
        GSM8K_FILENAME if dataset_choice.startswith("GSM8K") else SVAMP_FILENAME,
        language="text",
    )

    max_words = st.slider(
        "Maks kata distilasi Latent CoT", min_value=5, max_value=30, value=20,
        help="Jumlah maksimum kata yang dipertahankan saat answer distillation "
             "jika core answer bukan angka murni.",
    )

    st.divider()
    run_btn = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)
    st.caption(
        "Catatan: file dataset harus berada di folder/repo yang sama dengan "
        "`app.py`. Jika tidak ditemukan, ubah `GSM8K_FILENAME` / `SVAMP_FILENAME` "
        "di bagian atas `app.py`."
    )


# =========================================================
# 10. MAIN — LOAD & ANALYZE
# =========================================================

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

if run_btn:
    with st.spinner("Memuat dataset..."):
        if dataset_choice.startswith("GSM8K"):
            norm_df, err = load_gsm8k(GSM8K_PATH)
        else:
            norm_df, err = load_svamp(SVAMP_PATH)

    if err:
        st.error(f"❌ Gagal memuat dataset: {err}")
        st.info(
            f"File `{GSM8K_FILENAME if dataset_choice.startswith('GSM8K') else SVAMP_FILENAME}` "
            "tidak ditemukan di folder yang sama dengan `app.py`. Jika di-deploy dari GitHub, "
            "pastikan file dataset ikut di-commit ke repo pada folder yang sama (atau sesuaikan "
            "`GSM8K_FILENAME` / `SVAMP_FILENAME` di bagian atas `app.py` jika ada di subfolder)."
        )
        st.session_state.analyzed = False
    else:
        with st.spinner("Menjalankan ekstraksi jawaban, distilasi, dan penghitungan token..."):
            df, results, accuracy_results = run_analysis(norm_df, max_words)
            eff = compute_efficiency(results)

        st.session_state.df = df
        st.session_state.results = results
        st.session_state.accuracy_results = accuracy_results
        st.session_state.eff = eff
        st.session_state.dataset_name = "GSM8K-style" if dataset_choice.startswith("GSM8K") else "SVAMP"
        st.session_state.analyzed = True


if not st.session_state.analyzed:
    st.info("👈 Pilih dataset di sidebar, lalu klik **Jalankan Analisis** untuk memulai.")
    st.stop()


# =========================================================
# 11. RETRIEVE STATE
# =========================================================

df = st.session_state.df
results = st.session_state.results
accuracy_results = st.session_state.accuracy_results
eff = st.session_state.eff
dataset_name = st.session_state.dataset_name

st.success(f"✅ Analisis selesai — Dataset: **{dataset_name}** ({len(df)} sampel)")

tab_overview, tab_token, tab_accuracy, tab_distribusi, tab_data = st.tabs(
    ["📊 Ringkasan", "📉 Efisiensi Token", "🎯 Akurasi", "📈 Distribusi", "🗂️ Data"]
)


# =========================================================
# TAB 1: RINGKASAN
# =========================================================

with tab_overview:
    st.subheader("Ringkasan Utama")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rata-rata Token — No CoT", f"{eff['no_cot_mean']:.1f}")
    c2.metric("Rata-rata Token — Full CoT", f"{eff['cot_mean']:.1f}",
              delta=f"+{eff['overhead_cot']:.1f}% vs No CoT", delta_color="inverse")
    c3.metric("Rata-rata Token — Latent CoT", f"{eff['latent_mean']:.1f}",
              delta=f"-{eff['reduction_cot_lat']:.1f}% vs Full CoT", delta_color="normal")

    st.divider()

    c4, c5, c6 = st.columns(3)
    c4.metric("Akurasi No CoT", f"{accuracy_results['no_cot']['accuracy']:.2f}%")
    c5.metric("Akurasi Full CoT", f"{accuracy_results['cot']['accuracy']:.2f}%")
    c6.metric("Akurasi Latent CoT", f"{accuracy_results['latent_cot']['accuracy']:.2f}%")

    st.divider()

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**💾 Reduksi Total Token**")
        st.write(
            f"- Total token Full CoT: **{eff['total_cot']:,}**\n"
            f"- Total token Latent CoT: **{eff['total_latent']:,}**\n"
            f"- Reduksi total: **-{eff['total_reduction']:.2f}%**"
        )
    with colB:
        st.markdown("**🧮 Skenario Kompresi Teoritis**")
        rows = []
        for cr in [0.10, 0.20, 0.30]:
            compressed = eff["cot_mean"] * cr
            rows.append({"Compression Ratio": f"{cr:.0%}",
                          "Avg Token (proyeksi)": round(compressed, 1),
                          "Hemat vs Full CoT": f"{(1 - cr) * 100:.0f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**📋 Tabel Ringkasan (Paper-Ready)**")
    summary_table = pd.DataFrame({
        "Method": ["No CoT", "Full CoT", "Latent CoT"],
        "Avg Tokens": [results[m]["mean"] for m in MODES],
        "Median": [results[m]["median"] for m in MODES],
        "Min": [results[m]["min"] for m in MODES],
        "Max": [results[m]["max"] for m in MODES],
        "Total Tokens": [results[m]["total"] for m in MODES],
        "Accuracy (%)": [accuracy_results[m]["accuracy"] for m in MODES],
    })
    st.dataframe(
        summary_table.style.format({
            "Avg Tokens": "{:.2f}", "Median": "{:.1f}",
            "Accuracy (%)": "{:.2f}", "Total Tokens": "{:,}",
        }),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇️ Unduh Tabel Ringkasan (CSV)",
        summary_table.to_csv(index=False).encode("utf-8"),
        file_name=f"summary_table_{dataset_name}.csv",
        mime="text/csv",
    )


# =========================================================
# TAB 2: EFISIENSI TOKEN
# =========================================================

with tab_token:
    st.subheader("Efisiensi Penggunaan Token")

    fig1, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig1.patch.set_facecolor(DARK_BG)

    means = [eff["no_cot_mean"], eff["cot_mean"], eff["latent_mean"]]
    bars = axes[0].bar(MODE_LABELS, means, color=COLORS, width=0.5, edgecolor="none", zorder=3)
    for bar, val in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center", va="bottom", color=WHITE,
                     fontsize=9, fontweight="bold")
    axes[0].annotate(
        f"−{eff['reduction_cot_lat']:.1f}%",
        xy=(2, eff["latent_mean"]), xytext=(1.5, (eff["cot_mean"] + eff["latent_mean"]) / 2),
        arrowprops=dict(arrowstyle="->", color="#2dd68b", lw=1.5),
        fontsize=10, color="#2dd68b", fontweight="bold",
    )
    dark_ax(axes[0], "Rata-rata Token per Mode", ylabel="Token (rata-rata)")
    axes[0].set_ylim(0, max(means) * 1.25)

    data_bp = [results[m]["series"].values for m in MODES]
    bp = axes[1].boxplot(
        data_bp, tick_labels=["No CoT", "Full CoT", "Latent\nCoT"], patch_artist=True,
        medianprops=dict(color=WHITE, linewidth=2),
        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
        flierprops=dict(markerfacecolor=MUTED, marker="o", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    dark_ax(axes[1], "Distribusi Token per Mode", ylabel="Token")

    for mode, color, label in zip(MODES, COLORS, MODE_LABELS_SHORT):
        sd = np.sort(results[mode]["series"].values)
        yv = np.arange(len(sd)) / float(len(sd))
        axes[2].plot(sd, yv, color=color, label=label, linewidth=2)
    dark_ax(axes[2], "CDF Token Usage", xlabel="Tokens", ylabel="CDF")
    axes[2].legend(labelcolor=WHITE, framealpha=0.2, fontsize=8)

    patches_leg = [mpatches.Patch(color=c, label=l.replace("\n", " "))
                   for c, l in zip(COLORS, MODE_LABELS)]
    fig1.legend(handles=patches_leg, loc="lower center", ncol=3,
                framealpha=0.15, labelcolor=WHITE, fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    st.pyplot(fig1)
    plt.close(fig1)

    st.divider()

    fig3, ax3 = plt.subplots(figsize=(7, 4.2))
    fig3.patch.set_facecolor(DARK_BG)
    methods = ["Full CoT", "Latent CoT"]
    tokens = [eff["cot_mean"], eff["latent_mean"]]
    bars3 = ax3.bar(methods, tokens, color=[COLORS[1], COLORS[2]], edgecolor="none",
                     zorder=3, width=0.45)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f"{yval:.1f}",
                 ha="center", va="bottom", color=WHITE, fontsize=10, fontweight="bold")
    dark_ax(ax3, "Token Usage Reduction: Full CoT vs Latent CoT", ylabel="Average Tokens")
    ax3.annotate(
        f"−{eff['reduction_cot_lat']:.1f}%",
        xy=(1, eff["latent_mean"]), xytext=(0.4, (eff["cot_mean"] + eff["latent_mean"]) / 2),
        arrowprops=dict(arrowstyle="->", color="#2dd68b", lw=1.5),
        fontsize=10, color="#2dd68b", fontweight="bold",
    )
    st.pyplot(fig3)
    plt.close(fig3)

    st.info(
        f"**Overhead** No CoT → Full CoT: `+{eff['overhead_cot']:.2f}%`  |  "
        f"**Reduksi** Full CoT → Latent CoT: `-{eff['reduction_cot_lat']:.2f}%`  |  "
        f"**Total token dihemat**: `{eff['total_cot'] - eff['total_latent']:,}` token "
        f"(`-{eff['total_reduction']:.2f}%`)"
    )


# =========================================================
# TAB 3: AKURASI
# =========================================================

with tab_accuracy:
    st.subheader("Analisis Akurasi")

    acc_values = [accuracy_results[m]["accuracy"] for m in MODES]

    fig_acc, ax_acc = plt.subplots(figsize=(8, 4.5))
    fig_acc.patch.set_facecolor(DARK_BG)
    acc_bars = ax_acc.bar(MODE_LABELS, acc_values, color=COLORS, width=0.55,
                          edgecolor="none", zorder=3)
    for bar, value in zip(acc_bars, acc_values):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, min(bar.get_height() + 1.5, 99.5),
                    f"{value:.1f}%", ha="center", va="bottom", color=WHITE,
                    fontsize=10, fontweight="bold")
    dark_ax(ax_acc, "Akurasi per Metode (Soft-Match)", ylabel="Akurasi (%)")
    ax_acc.set_ylim(0, max(100, max(acc_values) + 5))
    st.pyplot(fig_acc)
    plt.close(fig_acc)

    st.caption(
        "Soft-match: cocok persis setelah normalisasi angka, atau selisih ≤1 untuk "
        "jawaban integer (toleransi pembulatan)."
    )

    st.divider()

    if "Type" in df.columns and df["Type"].nunique() > 1:
        st.markdown("**Akurasi Latent CoT per Tipe Soal**")
        type_acc = (
            df.groupby("Type")["correct_latent_cot"].mean().sort_values() * 100
        )
        fig_type, ax_type = plt.subplots(figsize=(8, max(3, len(type_acc) * 0.5)))
        fig_type.patch.set_facecolor(DARK_BG)
        pal = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(type_acc)))
        hbars = ax_type.barh(type_acc.index, type_acc.values, color=pal,
                             edgecolor="none", zorder=3)
        for bar, val in zip(hbars, type_acc.values):
            ax_type.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                        va="center", color=WHITE, fontsize=8, fontweight="bold")
        dark_ax(ax_type, "Latent CoT Accuracy per Tipe Soal", xlabel="Accuracy (%)", grid_axis="x")
        ax_type.set_xlim(0, 115)
        st.pyplot(fig_type)
        plt.close(fig_type)

    st.divider()
    st.markdown("**Correct vs Incorrect per Mode**")
    n_total = len(df)
    correct_counts = [accuracy_results[m]["correct"] for m in MODES]
    incorrect_counts = [n_total - c for c in correct_counts]

    fig_stack, ax_stack = plt.subplots(figsize=(7, 4.2))
    fig_stack.patch.set_facecolor(DARK_BG)
    x = np.arange(3)
    b_corr = ax_stack.bar(x, correct_counts, color="#2dd68b", edgecolor="none",
                          zorder=3, label="Correct")
    ax_stack.bar(x, incorrect_counts, bottom=correct_counts, color="#e05c5c",
                edgecolor="none", zorder=3, label="Incorrect")
    for bar, val in zip(b_corr, correct_counts):
        pct = val / n_total * 100
        ax_stack.text(bar.get_x() + bar.get_width() / 2, val / 2, f"{pct:.1f}%",
                      ha="center", va="center", color=WHITE, fontsize=9, fontweight="bold")
    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(MODE_LABELS_SHORT)
    dark_ax(ax_stack, "Correct vs Incorrect per Mode", ylabel="Jumlah Sampel")
    ax_stack.legend(labelcolor=WHITE, framealpha=0.2, fontsize=8)
    st.pyplot(fig_stack)
    plt.close(fig_stack)


# =========================================================
# TAB 4: DISTRIBUSI
# =========================================================

with tab_distribusi:
    st.subheader("Distribusi Jumlah Token")

    fig2, ax2 = plt.subplots(figsize=(10, 4.2))
    fig2.patch.set_facecolor(DARK_BG)
    for mode, color, label in zip(MODES, COLORS,
                                   ["No CoT", "Full CoT (Standard)", "Latent CoT (Proposed)"]):
        ax2.hist(results[mode]["series"], bins=40, alpha=0.65, color=color,
                 label=label, edgecolor="none")
    dark_ax(ax2, "Distribusi Jumlah Token per Mode", xlabel="Jumlah Token", ylabel="Frekuensi")
    ax2.legend(labelcolor=WHITE, framealpha=0.2)
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()
    st.markdown("**Statistik Token per Kolom Dasar**")
    base_stats = pd.DataFrame({
        "Kolom": ["Question", "Answer", "Reasoning/Equation"],
        "Rata-rata Token": [
            df["question_tokens"].mean(),
            df["answer_tokens"].mean(),
            df["reasoning_tokens"].mean(),
        ],
    })
    st.dataframe(
        base_stats.style.format({"Rata-rata Token": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )


# =========================================================
# TAB 5: DATA
# =========================================================

with tab_data:
    st.subheader("Data Hasil Analisis")
    show_cols = [
        "problem", "answer_str", "tokens_no_cot", "tokens_cot", "tokens_latent_cot",
        "correct_no_cot", "correct_cot", "correct_latent_cot",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, height=420)

    st.download_button(
        "⬇️ Unduh Data Lengkap (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"hasil_token_analysis_{dataset_name}.csv",
        mime="text/csv",
    )

    with st.expander("🔍 Lihat contoh output per mode (5 sampel pertama)"):
        for mode in MODES:
            st.markdown(f"**{mode.upper()}**")
            for txt in results[mode]["outputs"].head(5):
                st.code(txt, language="text")