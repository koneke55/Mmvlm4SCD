"""Generate a LaTeX version of the paper from the live JSON artefacts.

Outputs:
    paper/paper.tex
    paper/references.bib
    paper/Makefile (only if missing)

The LaTeX file mirrors the ReportLab PDF produced by ``build_paper.py``;
its numbers come from the same ``experiments/results/`` JSON files so
both renderings stay in sync.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    with path.open() as f:
        return json.load(f)


def _safe_load(path: Path):
    return _load(path) if path.exists() else None


def _meanstd(vals):
    if not vals:
        return float("nan"), 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(mean(vals)), float(pstdev(vals))


def _fmt(m, s, d=3):
    if m != m:
        return "--"
    if s == 0:
        return f"{m:.{d}f}"
    return f"${m:.{d}f} \\pm {s:.{d}f}$"


def _esc(s: str) -> str:
    return (s.replace("\\", "\\\\").replace("&", "\\&").replace("%", "\\%")
             .replace("$", "\\$").replace("#", "\\#").replace("_", "\\_")
             .replace("{", "\\{").replace("}", "\\}"))


def build():
    out_tex = ROOT / "paper" / "paper.tex"
    out_bib = ROOT / "paper" / "references.bib"

    base = ROOT / "experiments" / "results" / "mmvlm4scd_default"
    summary = _load(base / "summary.json")
    ablation = _load(base / "per_modality_ablation.json")
    importance = _load(base / "modality_importance.json")
    bdir = ROOT / "experiments" / "results" / "baselines"
    logreg = _load(bdir / "logreg_severity.json")
    cox = _load(bdir / "cox_survival.json")
    fusion = _safe_load(ROOT / "experiments" / "results" /
                        "fusion_comparison" / "per_fusion.json")
    subgroup = _safe_load(ROOT / "experiments" / "results" /
                          "subgroups" / "subgroups.json")
    brier = _safe_load(ROOT / "experiments" / "results" /
                       "survival_horizons" / "brier.json")

    aurocs = [r["auroc_ovr"] for r in summary["test_per_seed"]]
    accs = [r["accuracy"] for r in summary["test_per_seed"]]
    f1s = [r["f1_macro"] for r in summary["test_per_seed"]]
    cidxs = [r["c_index"] for r in summary["test_per_seed"]]

    n_patients = summary["n_patients"]
    n_seeds = summary["seeds"]
    auroc_m, auroc_s = _meanstd(aurocs)
    acc_m, acc_s = _meanstd(accs)
    f1_m, f1_s = _meanstd(f1s)
    c_m, c_s = _meanstd(cidxs)

    fig_base = "../experiments/results/mmvlm4scd_default/figures"

    L: list[str] = []
    a = L.append

    # ----------------------------- preamble -------------------------------
    a(r"% !TeX program = pdflatex")
    a(r"\documentclass[10pt,twocolumn,letterpaper]{article}")
    a(r"\usepackage[utf8]{inputenc}")
    a(r"\usepackage[T1]{fontenc}")
    a(r"\usepackage[letterpaper,margin=0.75in]{geometry}")
    a(r"\usepackage{microtype}")
    a(r"\usepackage{graphicx}")
    a(r"\usepackage{booktabs}")
    a(r"\usepackage{array}")
    a(r"\usepackage{caption}")
    a(r"\usepackage{xcolor}")
    a(r"\usepackage[hidelinks]{hyperref}")
    a(r"\usepackage{authblk}")
    a(r"\usepackage{abstract}")
    a(r"\usepackage{amsmath}")
    a(r"\usepackage{amssymb}")
    a(r"\usepackage{textcomp}")
    a(r"\usepackage{enumitem}")
    a(r"\usepackage{cite}")
    a(r"\setlength{\columnsep}{0.3in}")
    a(r"\definecolor{accent}{HTML}{1A3D63}")
    a(r"\renewcommand{\abstractname}{\textcolor{accent}{Abstract}}")
    a(r"\captionsetup{font=small,labelfont=bf}")
    a(r"\newcommand{\headcolor}[1]{\textcolor{accent}{#1}}")
    a("")
    a(r"\title{Multimodal Modeling of Sickle Cell Disease:\\ "
      r"A Quantitative Study of Severity Stratification and Survival Prediction}")
    a(r"\author{Sambou Kone}")
    a(r"\affil{Independent researcher \textperiodcentered{} Mmvlm4SCD project "
      r"\textperiodcentered{} \href{https://github.com/koneke55/Mmvlm4SCD}"
      r"{github.com/koneke55/Mmvlm4SCD}}")
    a(r"\date{" + date.today().isoformat() + r"}")
    a("")
    a(r"\begin{document}")
    a(r"\twocolumn[\maketitle\begin{onecolabstract}")

    # ------------------------------ abstract ------------------------------
    a("Sickle Cell Disease (SCD) is a monogenic but phenotypically heterogeneous "
      "disorder for which clinical decisions hinge on stratifying patients by "
      "severity and projecting long-term survival. We present "
      "\\textbf{Mmvlm4SCD}, an open multimodal deep-learning framework that "
      "fuses clinical, genomic, peripheral-blood-smear imaging and longitudinal "
      "vitals/labs trajectories into a shared embedding optimised jointly for "
      "ordinal severity classification and Cox proportional-hazards survival "
      f"prediction. Across {n_patients} simulated SCD patients calibrated to "
      "published haematological literature and "
      f"{n_seeds} random initialisations, the multimodal attention-fusion "
      f"model reaches a macro AUROC of \\textbf{{{_fmt(auroc_m, auroc_s)}}} for "
      "severity (mild/moderate/severe) and a Harrell C-index of "
      f"\\textbf{{{_fmt(c_m, c_s)}}} for time-to-death. A clinical-only "
      f"logistic-regression baseline reaches AUROC = {logreg['auroc_ovr']:.3f} "
      f"and a clinical-only Cox model reaches C-index = {cox['c_index']:.3f}, "
      "showing that on this benchmark the marginal lift from non-clinical "
      "modalities is small but mostly beneficial when paired with attention "
      "fusion. Modality-ablation analysis confirms clinical features dominate "
      "while imaging and genomic streams contribute complementary signal. We "
      "release the code, the synthetic-cohort generator, the data registry of "
      "public SCD sources, and a reproducible experimental pipeline to support "
      "extension to credentialed cohorts (dbGaP, MIMIC-IV, UK Biobank, NHLBI "
      "CuRe-SCD).")
    a("")
    a(r"\textbf{Keywords:} sickle cell disease \textperiodcentered{} "
      r"multimodal learning \textperiodcentered{} survival analysis "
      r"\textperiodcentered{} severity stratification \textperiodcentered{} "
      r"clinical AI \textperiodcentered{} reproducible research")
    a(r"\end{onecolabstract}\bigskip]")

    # ------------------------------ 1. Intro ------------------------------
    a(r"\section{Introduction}")
    a("Sickle Cell Disease (SCD) is the most common monogenic disorder "
      "worldwide, caused by a single beta-globin missense mutation "
      "(\\textit{HBB} p.Glu6Val) that polymerises haemoglobin S under "
      "deoxygenation, deforms erythrocytes and triggers chronic haemolysis, "
      "vaso-occlusion, end-organ damage and shortened life expectancy "
      "\\cite{Piel2017,GBD2021SCD}. Despite a single causal mutation, the "
      "clinical course varies dramatically: patients with the same genotype "
      "can experience anywhere from $<\\!0.5$ to $>\\!10$ vaso-occlusive "
      "crises per year, and median survival differs by more than two decades "
      "across subgroups \\cite{Piel2017}.")
    a("")
    a("Modern SCD care produces heterogeneous longitudinal data: complete "
      "blood counts, biochemistry panels, ICU and ED encounters, peripheral-"
      "blood-smear microscopy, genome-wide modifier variants, and patient-"
      "reported pain trajectories. Existing risk scores compress this "
      "complexity into hand-crafted formulae \\cite{Sebastiani2010,Quinn2007}. "
      "We hypothesise that a \\emph{jointly trained} multimodal model can "
      "recover stronger, calibrated decision functions for two clinically "
      "pivotal tasks: \\textbf{(i)} ordinal severity stratification "
      "(mild/moderate/severe) and \\textbf{(ii)} long-horizon survival "
      "prediction.")
    a("")
    a("This paper makes three contributions. \\textbf{(1)} An open, "
      "reproducible multimodal architecture --- \\textit{Mmvlm4SCD} --- with "
      "pluggable encoders for clinical, genomic, imaging and temporal data, "
      "three fusion strategies, and a dual-task head trained with cross-"
      "entropy + Cox partial likelihood. \\textbf{(2)} A literature-calibrated "
      "synthetic cohort generator that allows benchmarking before credential-"
      "gated cohorts are obtained, plus a curated registry of 14 public SCD "
      "data sources spanning imaging, transcriptomics, ICU labs and "
      "population genetics. \\textbf{(3)} A quantitative study comparing the "
      "multimodal model against strong tabular baselines, with seed-replicated "
      "metrics, modality-ablation, gradient-based interpretability, fusion-"
      "strategy comparison, bootstrap CIs, time-dependent Brier scores and "
      "subgroup analysis.")

    # ---------------------------- 2. Background ---------------------------
    a(r"\section{Background and Related Work}")
    a("\\textbf{SCD severity scoring.} Sebastiani et al. derived a Bayesian "
      "network severity score for SCD using clinical and laboratory variables "
      "\\cite{Sebastiani2010}; Quinn et al. quantified early-life predictors "
      "of severe disease \\cite{Quinn2007}. These remain dominant baselines "
      "in clinical practice but do not incorporate imaging or molecular "
      "data.")
    a("")
    a("\\textbf{Survival modelling.} The Cox proportional-hazards model "
      "\\cite{Cox1972} is the de-facto SCD survival baseline. Deep-learning "
      "extensions such as DeepSurv \\cite{Katzman2018} retain the partial-"
      "likelihood objective while substituting a neural risk function for the "
      "linear predictor. We adopt this objective for the survival head.")
    a("")
    a("\\textbf{Multimodal medical AI.} Cross-attention and Transformer-based "
      "fusion across imaging, genomics and tabular EHR have shown gains in "
      "oncology and ophthalmology \\cite{Acosta2022,Steyaert2023}. Application "
      "to SCD has been limited by data fragmentation; our registry "
      "consolidates the available open and credentialed sources to lower this "
      "barrier.")
    a("")
    a("\\textbf{Imaging.} Convolutional networks readily distinguish sickled "
      "erythrocytes from normal cells on peripheral-blood-smear images "
      "\\cite{Xu2017,Alzubaidi2020}; we treat such CNN outputs as imaging "
      "embeddings within fusion.")

    # ----------------------------- 3. Data --------------------------------
    a(r"\section{Data}")
    a(r"\subsection{Public SCD data registry}")
    a("We catalogue 14 public SCD-relevant data sources covering clinical, "
      "genomic, imaging and epidemiological modalities (Table~\\ref{tab:registry}). "
      "The canonical machine-readable list lives in "
      "\\texttt{src/mmvlm4scd/data/registry.py}.")
    a("")
    a(r"\begin{table}[t]\centering")
    a(r"\caption{Public SCD data sources tracked by the Mmvlm4SCD registry. "
      r"Access tiers: \textit{open}, \textit{registered}, \textit{dua-required} "
      r"(requires Data Use Agreement).}\label{tab:registry}")
    a(r"\small")
    a(r"\begin{tabular}{p{4.2cm}lll}")
    a(r"\toprule")
    a(r"Source & Modality & Access \\")
    a(r"\midrule")
    sources = [
        ("NHLBI CuRe-SCD Data Hub", "multimodal", "registered"),
        ("dbGaP phs001514 (SCDIC Registry)", "clinical", "dua-required"),
        ("dbGaP phs001599 (Walk-PHaSST)", "clinical", "dua-required"),
        ("GEO GSE53441 (SCD whole-blood)", "genomic", "open"),
        ("GEO GSE35007 (pediatric SCD)", "genomic", "open"),
        ("MIMIC-IV (D57.* subset)", "clinical", "dua-required"),
        ("UK Biobank SCD subset", "multimodal", "dua-required"),
        ("erythrocytesIDB", "imaging", "open"),
        ("Kaggle Sickle RBC", "imaging", "open"),
        ("ClinVar -- HBB variants", "genomic", "open"),
        ("gnomAD v4 -- HBB", "genomic", "open"),
        ("NHLBI TOPMed", "genomic", "dua-required"),
        ("WHO SCD country profiles", "clinical", "open"),
        ("GBD 2021 SCD", "clinical", "open"),
    ]
    for name, mod, acc in sources:
        a(f"{_esc(name)} & {mod} & {acc} \\\\")
    a(r"\bottomrule\end{tabular}\end{table}")

    a(r"\subsection{Synthetic cohort}")
    a("Because dbGaP, UK Biobank and MIMIC-IV require Data Use Agreements that "
      "we did not have access to during this study, all empirical results in "
      "this paper are computed on a \\textbf{literature-calibrated synthetic "
      "cohort} shipped with the code base. The generator draws "
      f"{n_patients} virtual patients with marginal distributions calibrated to: SCD "
      "genotype prevalences (HbSS, HbSC, HbS$\\beta^{+}$, HbS$\\beta^{0}$); "
      "clinical labs (Hb, HbF\\%, WBC, platelets, LDH, total bilirubin, CRP); "
      "annual vaso-occlusive crisis rates; acute chest syndrome and stroke "
      "history; HBB and modifier variant indicators; CNN-style imaging "
      "embeddings of sickled vs normal erythrocytes; and monthly trajectories "
      "of vitals and pain VAS. Severity labels are derived as quantiles of a "
      "weighted clinical/genotype score; survival times are sampled from a "
      "Weibull model with severity- and genotype-dependent hazard.")
    a("")
    a("\\textbf{Synthetic-data disclaimer.} The cohort is not real patient "
      "data. Results should be read as architectural benchmarks, not clinical "
      "claims. The codebase is designed so that, once an investigator obtains "
      "DUA-gated cohorts, the same training and evaluation entry points can "
      "be invoked with minimal modification to the data loader.")

    # ---------------------------- 4. Methods ------------------------------
    a(r"\section{Methods}")
    a(r"\subsection{Architecture}")
    a("Mmvlm4SCD comprises four modality-specific encoders, a fusion module, "
      "and two prediction heads. The clinical encoder is a 3-layer MLP over "
      "standardised tabular features. The genomic encoder splits the input "
      "into a binary HBB/modifier-variant block and a continuous polygenic-"
      "like block. The imaging encoder consumes pre-computed CNN embeddings. "
      "The temporal encoder is a one-layer GRU over monthly vitals/labs "
      "trajectories. All four encoders project to a shared 64-dimensional "
      "embedding (default).")
    a("")
    a("Three fusion strategies are implemented and benchmarked: "
      "\\textbf{attention} (Transformer encoder over modality tokens with a "
      "learned [CLS] readout), \\textbf{cross-attention} (clinical embedding "
      "queries the other modalities), and \\textbf{late fusion} (concatenate "
      "then project).")

    a(r"\subsection{Training objective}")
    a("Severity uses multinomial cross-entropy. Survival uses the negative "
      "Breslow partial log-likelihood:")
    a(r"\begin{equation}")
    a(r"\mathcal{L}_{\mathrm{cox}} = -\frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}}"
      r"\!\left[r_i - \log\!\sum_{j: t_j\ge t_i}\exp(r_j)\right],")
    a(r"\end{equation}")
    a("where $r_i$ is the predicted risk score, $t_i$ the observed time, and "
      "$\\mathcal{D}$ the set of subjects with an observed event. The combined "
      "objective is $\\mathcal{L}=\\alpha\\,\\mathcal{L}_\\mathrm{ce}+\\beta\\,"
      "\\mathcal{L}_\\mathrm{cox}$ with default $(\\alpha,\\beta)=(1.0,0.5)$. "
      "Optimisation uses AdamW with cosine annealing and gradient clipping at "
      "norm 1.0. Validation AUROC drives early stopping.")

    a(r"\subsection{Evaluation}")
    a("We report \\textbf{accuracy}, \\textbf{macro F1} and \\textbf{macro "
      "one-vs-rest AUROC} for severity; \\textbf{Harrell's C-index} "
      "\\cite{Harrell1996}, time-dependent Brier scores and the Integrated "
      "Brier Score (IBS) following Graf et al.~\\cite{Graf1999} for survival. "
      "We additionally report calibration via reliability diagrams, modality "
      "importance via gradient L2 norms, and Kaplan-Meier curves stratified "
      f"by predicted-risk tertile. Results are reported as mean $\\pm$ "
      f"standard deviation across {n_seeds} random seeds.")

    # --------------------------- 5. Experiments ---------------------------
    a(r"\section{Experiments}")
    a(r"\subsection{Setup}")
    a(f"Cohort: {n_patients} synthetic patients; 70/15/15 train/val/test "
      "split; batch size 64; 30 epochs with early-stop patience 8. Hardware: "
      "CPU-only PyTorch 2.x. Code at "
      "\\url{https://github.com/koneke55/Mmvlm4SCD}. Seeds $\\{0,1,2\\}$; "
      "configs in \\texttt{configs/}.")

    a(r"\subsection{Main results}")
    a(r"\begin{table}[t]\centering")
    a(r"\caption{Held-out test performance over " + str(n_seeds) +
      r" seeds. Mmvlm4SCD scores reported as mean $\pm$ standard deviation.}"
      r"\label{tab:main}")
    a(r"\small")
    a(r"\begin{tabular}{lcccc}")
    a(r"\toprule")
    a(r"Model & Acc & F1 & AUROC & C-index \\")
    a(r"\midrule")
    a(f"LR (clinical) & {logreg['accuracy']:.3f} & {logreg['f1_macro']:.3f} & "
      f"{logreg['auroc_ovr']:.3f} & -- \\\\")
    a(f"Cox PH (clinical) & -- & -- & -- & {cox['c_index']:.3f} \\\\")
    a("Mmvlm4SCD (attention) & "
      f"{_fmt(acc_m, acc_s)} & {_fmt(f1_m, f1_s)} & "
      f"{_fmt(auroc_m, auroc_s)} & {_fmt(c_m, c_s)} \\\\")
    a(r"\bottomrule\end{tabular}\end{table}")

    # Per-modality ablation
    a(r"\subsection{Per-modality ablation}")
    a(r"\begin{table}[t]\centering")
    a(r"\caption{Test performance when each modality is set to zero at train "
      r"and test time.}\label{tab:ablation}")
    a(r"\small")
    a(r"\begin{tabular}{lcccc}")
    a(r"\toprule")
    a(r"Dropped & Acc & F1 & AUROC & C-index \\")
    a(r"\midrule")
    for k, m in ablation.items():
        a(f"{_esc(k)} & {m['accuracy']:.3f} & {m['f1_macro']:.3f} & "
          f"{m['auroc_ovr']:.3f} & {m['c_index']:.3f} \\\\")
    a(r"\bottomrule\end{tabular}\end{table}")

    # Bootstrap
    if subgroup is not None:
        bs = subgroup["bootstrap_overall"]
        a(r"\subsubsection{Bootstrap confidence intervals}")
        a("Non-parametric 95\\% bootstrap CIs over 300 resamples of the "
          "held-out test set for the attention-fusion model:")
        a(r"\begin{table}[t]\centering")
        a(r"\caption{Bootstrap CIs for the test metrics.}\label{tab:boot}")
        a(r"\small")
        a(r"\begin{tabular}{lccc}")
        a(r"\toprule")
        a(r"Metric & Mean & 95\% CI low & 95\% CI high \\")
        a(r"\midrule")
        for nice, k in [("Accuracy", "accuracy"),
                        ("F1 (macro)", "f1_macro"),
                        ("AUROC (OvR)", "auroc_ovr"),
                        ("C-index", "c_index")]:
            a(f"{nice} & {bs[k]['mean']:.3f} & "
              f"{bs[k]['ci_low']:.3f} & {bs[k]['ci_high']:.3f} \\\\")
        a(r"\bottomrule\end{tabular}\end{table}")

    # Fusion comparison
    if fusion is not None:
        a(r"\subsubsection{Fusion-strategy comparison}")
        a(r"\begin{table}[t]\centering")
        a(r"\caption{Comparison of the three fusion strategies (3 seeds each)."
          r"}\label{tab:fusion}")
        a(r"\small")
        a(r"\begin{tabular}{lcccc}")
        a(r"\toprule")
        a(r"Fusion & AUROC & C-index & Acc & F1 \\")
        a(r"\midrule")
        for name in ("attention", "cross", "late"):
            d = fusion[name]
            a(f"{_esc(name)} & "
              f"${d['mean_auroc']:.3f}\\pm{d['std_auroc']:.3f}$ & "
              f"${d['mean_c_index']:.3f}\\pm{d['std_c_index']:.3f}$ & "
              f"{d['mean_accuracy']:.3f} & {d['mean_f1_macro']:.3f} \\\\")
        a(r"\bottomrule\end{tabular}\end{table}")
        a(r"\begin{figure}[t]\centering")
        a(r"\includegraphics[width=\linewidth]"
          r"{../experiments/results/fusion_comparison/fusion_bar.png}")
        a(r"\caption{Severity AUROC (left) and survival C-index (right) by "
          r"fusion strategy.}\label{fig:fusion}\end{figure}")

    # Modality importance
    a(r"\subsection{Modality importance}")
    a("We additionally measure modality importance by averaging the L2 norm "
      "of the gradient of the predicted-class severity logit with respect to "
      "each modality input over the test set "
      f"(Figure~\\ref{{fig:imp}}). Clinical features dominate "
      f"({importance['clinical']:.2f}), followed by imaging "
      f"({importance['imaging']:.2f}), genomic ({importance['genomic']:.2f}) "
      f"and temporal ({importance['temporal']:.2f}). Ablation and gradient "
      "analyses agree directionally.")

    # Standard figures
    a(r"\begin{figure}[t]\centering")
    a(r"\includegraphics[width=\linewidth]{" + fig_base +
      r"/training_curves.png}")
    a(r"\caption{Training loss and validation metrics (seed 0)."
      r"}\label{fig:train}\end{figure}")

    a(r"\begin{figure}[t]\centering")
    a(r"\includegraphics[width=0.85\linewidth]{" + fig_base +
      r"/confusion.png}")
    a(r"\caption{Severity confusion matrix on the held-out test set (seed 0)."
      r"}\label{fig:cm}\end{figure}")

    a(r"\begin{figure}[t]\centering")
    a(r"\includegraphics[width=0.9\linewidth]{" + fig_base +
      r"/modality_importance.png}")
    a(r"\caption{Per-modality gradient $L_{2}$ norm of the predicted severity "
      r"logit, averaged over the test set.}\label{fig:imp}\end{figure}")

    a(r"\begin{figure}[t]\centering")
    a(r"\includegraphics[width=\linewidth]{" + fig_base +
      r"/km_by_risk.png}")
    a(r"\caption{Kaplan-Meier survival curves stratified by predicted-risk "
      r"tertile from the Cox-trained survival head."
      r"}\label{fig:km}\end{figure}")

    a(r"\begin{figure}[t]\centering")
    a(r"\includegraphics[width=0.85\linewidth]{" + fig_base +
      r"/calibration.png}")
    a(r"\caption{Reliability diagram for the top-class severity probability."
      r"}\label{fig:cal}\end{figure}")

    # Survival horizons
    if brier is not None:
        a(r"\subsection{Time-dependent survival evaluation}")
        a("Beyond the Cox C-index we evaluate Brier scores at increasing "
          "follow-up horizons (1, 2, 5, 10, 15, 20 years) and the Integrated "
          "Brier Score (IBS) over the same interval. Brier(t) is computed "
          "using inverse-probability-of-censoring weighting "
          "\\cite{Graf1999}.")
        a(r"\begin{table}[t]\centering")
        a(r"\caption{Time-dependent Brier scores and IBS for the attention-"
          r"fusion model.}\label{tab:brier}")
        a(r"\small")
        cols = "l" + "c" * len(brier["horizons"]) + "c"
        a(r"\begin{tabular}{" + cols + r"}")
        a(r"\toprule")
        head = ["Horizon"] + [f"{h:g}" for h in brier["horizons"]] + ["IBS"]
        a(" & ".join(head) + r" \\")
        a(r"\midrule")
        row = ["Brier"] + [f"{brier['brier'][f'bs_{h:g}']:.3f}"
                           for h in brier["horizons"]] + [f"{brier['ibs']:.3f}"]
        a(" & ".join(row) + r" \\")
        a(r"\bottomrule\end{tabular}\end{table}")
        a(r"\begin{figure}[t]\centering")
        a(r"\includegraphics[width=0.9\linewidth]"
          r"{../experiments/results/survival_horizons/brier_curve.png}")
        a(r"\caption{Brier score over follow-up horizon with the IBS annotated."
          r"}\label{fig:brier}\end{figure}")

    # Subgroup analysis
    if subgroup is not None:
        a(r"\subsection{Subgroup analysis}")
        a(f"We slice the test set ($n={subgroup['n_test']}$) by genotype, age "
          "band and sex (Table~\\ref{tab:sg}, Figure~\\ref{fig:sg}). Subgroups "
          "with fewer than 20 patients are dropped to keep estimates stable. "
          f"Overall AUROC is {subgroup['overall']['auroc_ovr']:.3f} and "
          f"overall C-index is {subgroup['overall']['c_index']:.3f}.")
        a(r"\begin{table}[t]\centering")
        a(r"\caption{Subgroup-stratified test metrics.}\label{tab:sg}")
        a(r"\small")
        a(r"\begin{tabular}{lcc}")
        a(r"\toprule")
        a(r"Subgroup & AUROC & C-index \\")
        a(r"\midrule")
        for k, v in subgroup["groups"].items():
            au = v.get("auroc_ovr", float("nan"))
            ci = v.get("c_index", float("nan"))
            au_s = "--" if au != au else f"{au:.3f}"
            ci_s = "--" if ci != ci else f"{ci:.3f}"
            a(f"{_esc(k)} & {au_s} & {ci_s} \\\\")
        a(r"\bottomrule\end{tabular}\end{table}")
        a(r"\begin{figure}[t]\centering")
        a(r"\includegraphics[width=\linewidth]"
          r"{../experiments/results/subgroups/figures/subgroup_bars.png}")
        a(r"\caption{Subgroup AUROC (left) and C-index (right). Dashed "
          r"vertical lines mark the overall test-set value.}\label{fig:sg}"
          r"\end{figure}")

    # ---------------------------- 6. Discussion ---------------------------
    a(r"\section{Discussion}")
    a("Three observations emerge. \\textbf{(i)} A strong tabular baseline is "
      "hard to beat on this synthetic cohort. The logistic-regression "
      "severity classifier and the linear Cox model both perform "
      "competitively, indicating that the synthetic generator is --- by "
      "construction --- dominated by low-dimensional clinical signal. This "
      "motivates evaluating the multimodal architecture on cohorts with "
      "richer modalities, where genomic and imaging streams carry more "
      "\\emph{independent} information (e.g.~UK Biobank smears + TOPMed WGS). "
      "\\textbf{(ii)} The clinical modality is consistently the most "
      "informative according to both ablation and gradient analyses, "
      "mirroring clinical experience that haematological labs largely "
      "determine SCD prognosis. \\textbf{(iii)} Imaging contributes more than "
      "genomics to the model's decision-time signal in our setup; the "
      "synthetic imaging block encodes the sickled-morphology fraction (a "
      "direct phenotype) whereas the synthetic genomic block is closer to a "
      "mild risk modifier.")

    # ----------------------- 7. Limitations & ethics ----------------------
    a(r"\section{Limitations and Ethical Considerations}")
    a("Our experiments are run on simulated data. While marginal distributions "
      "are calibrated to SCD literature, the joint structure is parametric "
      "and necessarily under-represents the full heterogeneity of SCD. "
      "Results should not be interpreted as clinical evidence. Once we secure "
      "DUA approvals (target: dbGaP phs001514, MIMIC-IV-SCD subset, NHLBI "
      "CuRe-SCD), the same training pipeline can be invoked with replacement "
      "data loaders.")
    a("")
    a("Because SCD disproportionately affects populations of African ancestry, "
      "deployments must be audited for performance disparities by ancestry, "
      "age and sex; the codebase exposes subgroup-stratified evaluation "
      "hooks. We follow the Contributor Covenant Code of Conduct and release "
      "everything under MIT.")

    # -------------------------- 8. Reproducibility ------------------------
    a(r"\section{Reproducibility}")
    a("All numbers in this paper are produced by "
      "\\texttt{src/scripts/run\\_full\\_experiment.py --config "
      "configs/default.yaml --seeds 3 --ablate}, "
      "\\texttt{src/scripts/run\\_baselines.py}, "
      "\\texttt{src/scripts/run\\_fusion\\_comparison.py}, "
      "\\texttt{src/scripts/run\\_subgroup\\_analysis.py} and "
      "\\texttt{src/scripts/run\\_survival\\_horizons.py}, then rendered by "
      "\\texttt{paper/build\\_paper.py} (for the ReportLab PDF) or "
      "\\texttt{paper/build\\_tex.py} + \\texttt{pdflatex} (this document). "
      "JSON artefacts in \\texttt{experiments/results/} are the single source "
      "of truth.")

    # --------------------------- 9. Conclusion ----------------------------
    a(r"\section{Conclusion}")
    a("We presented Mmvlm4SCD, an open multimodal framework for Sickle Cell "
      "Disease that jointly learns severity stratification and survival "
      f"prediction. On a literature-calibrated synthetic benchmark, the "
      f"model attains AUROC = {_fmt(auroc_m, auroc_s)} and C-index = "
      f"{_fmt(c_m, c_s)} across seeds, with clinical features dominating "
      "decision-time importance. The released code, registry of public SCD "
      "sources, and reproducible pipeline lower the barrier to extending "
      "these experiments to real, credentialed cohorts --- the next step in "
      "establishing whether multimodal fusion delivers a clinically "
      "meaningful uplift over strong tabular baselines in SCD.")

    # ----------------------------- references -----------------------------
    a(r"\bibliographystyle{plain}")
    a(r"\bibliography{references}")
    a(r"\end{document}")

    out_tex.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ------------------------------- BibTeX -------------------------------
    bib = r"""@article{Piel2017,
  author = {Piel, F. B. and Steinberg, M. H. and Rees, D. C.},
  title = {Sickle cell disease},
  journal = {New England Journal of Medicine},
  volume = {376}, number = {16}, pages = {1561--1573}, year = {2017}}

@article{GBD2021SCD,
  author = {{GBD 2021 Sickle Cell Disease Collaborators}},
  title = {Global, regional and national prevalence and mortality burden of
           sickle cell disease, 2000--2021},
  journal = {The Lancet Haematology}, year = {2023}}

@article{Sebastiani2010,
  author = {Sebastiani, P. and Solovieff, N. and Hartley, S. W. and Milton, J. N.
            and Riva, A. and Dworkis, D. A. and Melista, E. and Klings, E. S. and
            Garrett, M. E. and Telen, M. J. and Ashley-Koch, A. and Baldwin, C. T.
            and Steinberg, M. H.},
  title = {A network model to predict the risk of death in sickle cell disease},
  journal = {Blood}, volume = {115}, number = {11}, pages = {2118--2127},
  year = {2010}}

@article{Quinn2007,
  author = {Quinn, C. T. and Rogers, Z. R. and McCavit, T. L. and Buchanan, G. R.},
  title = {Survival of children with sickle cell disease},
  journal = {Blood}, volume = {109}, number = {11}, pages = {4928--4933},
  year = {2007}}

@article{Cox1972,
  author = {Cox, D. R.},
  title = {Regression models and life-tables},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume = {34}, number = {2}, pages = {187--202}, year = {1972}}

@article{Katzman2018,
  author = {Katzman, J. L. and Shaham, U. and Cloninger, A. and Bates, J. and
            Jiang, T. and Kluger, Y.},
  title = {DeepSurv: personalized treatment recommender via Cox proportional
           hazards deep neural network},
  journal = {BMC Medical Research Methodology}, volume = {18}, pages = {24},
  year = {2018}}

@article{Acosta2022,
  author = {Acosta, J. N. and Falcone, G. J. and Rajpurkar, P. and Topol, E. J.},
  title = {Multimodal biomedical AI},
  journal = {Nature Medicine}, volume = {28}, number = {9},
  pages = {1773--1784}, year = {2022}}

@article{Steyaert2023,
  author = {Steyaert, S. and others},
  title = {Multimodal deep learning to predict prognosis in cancer},
  journal = {npj Precision Oncology}, volume = {7}, pages = {80}, year = {2023}}

@article{Xu2017,
  author = {Xu, M. and Papageorgiou, D. P. and Abidi, S. Z. and Dao, M. and
            Zhao, H. and Karniadakis, G. E.},
  title = {A deep convolutional neural network for classification of red blood
           cells in sickle cell anemia},
  journal = {PLOS Computational Biology}, volume = {13}, number = {10},
  pages = {e1005746}, year = {2017}}

@article{Alzubaidi2020,
  author = {Alzubaidi, L. and Fadhel, M. A. and Al-Shamma, O. and Zhang, J.
            and Duan, Y.},
  title = {Deep learning models for classification of red blood cells in
           microscopy images},
  journal = {Electronics}, volume = {9}, number = {3}, pages = {427},
  year = {2020}}

@article{Harrell1996,
  author = {Harrell, F. E. and Lee, K. L. and Mark, D. B.},
  title = {Multivariable prognostic models: issues in developing models,
           evaluating assumptions and adequacy, and measuring and reducing
           errors},
  journal = {Statistics in Medicine}, volume = {15}, number = {4},
  pages = {361--387}, year = {1996}}

@article{Graf1999,
  author = {Graf, E. and Schmoor, C. and Sauerbrei, W. and Schumacher, M.},
  title = {Assessment and comparison of prognostic classification schemes
           for survival data},
  journal = {Statistics in Medicine}, volume = {18}, number = {17--18},
  pages = {2529--2545}, year = {1999}}

@article{Karczewski2020,
  author = {Karczewski, K. J. and others},
  title = {The mutational constraint spectrum quantified from variation in
           141,456 humans},
  journal = {Nature}, volume = {581}, pages = {434--443}, year = {2020}}

@article{Johnson2023MIMICIV,
  author = {Johnson, A. E. W. and Bulgarelli, L. and Shen, L. and Gayles, A.
            and Shammout, A. and Horng, S. and Pollard, T. J. and Hao, S. and
            Moody, B. and Gow, B. and Lehman, L. H. and Celi, L. A. and Mark,
            R. G.},
  title = {MIMIC-IV, a freely accessible electronic health record dataset},
  journal = {Scientific Data}, volume = {10}, pages = {1}, year = {2023}}
"""
    out_bib.write_text(bib, encoding="utf-8")

    # ------------------------------ Makefile ------------------------------
    mk = ROOT / "paper" / "Makefile"
    if not mk.exists():
        mk.write_text(
            "PAPER=paper\n"
            "TEX=pdflatex -interaction=nonstopmode -halt-on-error\n\n"
            ".PHONY: all clean tectonic\n\n"
            "all: $(PAPER).pdf\n\n"
            "$(PAPER).pdf: $(PAPER).tex references.bib\n"
            "\t$(TEX) $(PAPER).tex\n"
            "\tbibtex $(PAPER)\n"
            "\t$(TEX) $(PAPER).tex\n"
            "\t$(TEX) $(PAPER).tex\n\n"
            "tectonic: $(PAPER).tex references.bib\n"
            "\ttectonic $(PAPER).tex\n\n"
            "clean:\n"
            "\trm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log "
            "$(PAPER).out $(PAPER).toc\n",
            encoding="utf-8")

    print(f"Wrote {out_tex} ({out_tex.stat().st_size/1024:.1f} KB)")
    print(f"Wrote {out_bib} ({out_bib.stat().st_size/1024:.1f} KB)")
    print(f"Wrote {mk}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.parse_args(argv)
    build()


if __name__ == "__main__":
    main()
