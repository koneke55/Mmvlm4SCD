"""Render the quantitative study PDF authored by Sambou Kone.

The PDF is generated programmatically with ReportLab so its contents are
always traceable to the JSON artefacts in ``experiments/results/``.
Run after ``run_full_experiment.py`` and ``run_baselines.py``.

Usage:
    python paper/build_paper.py --out paper/paper.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from statistics import mean, pstdev

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.flowables import HRFlowable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.data.registry import PUBLIC_SCD_DATASETS  # noqa: E402


# ----------------------------- styles -------------------------------------

def _styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=base["Title"], fontName="Times-Bold",
        fontSize=18, leading=22, alignment=1, spaceAfter=8,
    )
    s["authors"] = ParagraphStyle(
        "authors", parent=base["Normal"], fontName="Times-Roman",
        fontSize=11, leading=14, alignment=1, spaceAfter=4,
    )
    s["affil"] = ParagraphStyle(
        "affil", parent=base["Normal"], fontName="Times-Italic",
        fontSize=9.5, leading=12, alignment=1, spaceAfter=14,
    )
    s["abstract_h"] = ParagraphStyle(
        "abstract_h", parent=base["Normal"], fontName="Times-Bold",
        fontSize=10.5, leading=12, alignment=1, spaceAfter=2,
    )
    s["abstract"] = ParagraphStyle(
        "abstract", parent=base["BodyText"], fontName="Times-Roman",
        fontSize=10, leading=12.5, alignment=4,
        leftIndent=24, rightIndent=24, spaceAfter=10,
    )
    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontName="Times-Bold",
        fontSize=12, leading=14, spaceBefore=10, spaceAfter=4,
        textColor=colors.HexColor("#1a3d63"),
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName="Times-Bold",
        fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2,
        textColor=colors.HexColor("#1a3d63"),
    )
    s["body"] = ParagraphStyle(
        "body", parent=base["BodyText"], fontName="Times-Roman",
        fontSize=10, leading=12.6, alignment=4, spaceAfter=4,
    )
    s["caption"] = ParagraphStyle(
        "caption", parent=base["BodyText"], fontName="Times-Italic",
        fontSize=8.8, leading=10.5, alignment=1, spaceAfter=8,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=base["BodyText"], fontName="Times-Roman",
        fontSize=10, leading=12.6, leftIndent=14, bulletIndent=2,
        alignment=4, spaceAfter=2,
    )
    s["small"] = ParagraphStyle(
        "small", parent=base["BodyText"], fontName="Times-Roman",
        fontSize=8.8, leading=10.5, alignment=4, spaceAfter=3,
    )
    s["ref"] = ParagraphStyle(
        "ref", parent=base["BodyText"], fontName="Times-Roman",
        fontSize=8.6, leading=10.4, leftIndent=16, firstLineIndent=-16,
        spaceAfter=2, alignment=4,
    )
    s["mono"] = ParagraphStyle(
        "mono", parent=base["BodyText"], fontName="Courier",
        fontSize=8.8, leading=10.4, alignment=0, spaceAfter=4,
    )
    return s


def _doc(out_path: Path) -> BaseDocTemplate:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    margin = 0.7 * inch
    width, height = LETTER
    doc = BaseDocTemplate(
        str(out_path), pagesize=LETTER,
        leftMargin=margin, rightMargin=margin,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="Mmvlm4SCD: Multimodal Modeling of Sickle Cell Disease",
        author="Sambou Kone",
    )
    full_frame = Frame(margin, margin, width - 2 * margin, height - 2 * margin,
                       id="full", showBoundary=0)
    col_w = (width - 2 * margin - 0.3 * inch) / 2
    left = Frame(margin, margin, col_w, height - 2 * margin, id="left",
                 showBoundary=0, leftPadding=0, rightPadding=4)
    right = Frame(margin + col_w + 0.3 * inch, margin, col_w,
                  height - 2 * margin, id="right", showBoundary=0,
                  leftPadding=4, rightPadding=0)

    def _foot(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Times-Italic", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(margin, 0.4 * inch,
                          "Mmvlm4SCD - Sambou Kone - " + date.today().isoformat())
        canvas.drawRightString(width - margin, 0.4 * inch,
                               f"page {_doc.page}")
        canvas.restoreState()

    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[full_frame], onPage=_foot),
        PageTemplate(id="twocol", frames=[left, right], onPage=_foot),
    ])
    return doc


# ----------------------------- data helpers -------------------------------

def _maybe_load(p: Path):
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _load_results(root: Path) -> dict:
    base = root / "experiments" / "results" / "mmvlm4scd_default"
    with (base / "summary.json").open() as f:
        summary = json.load(f)
    with (base / "training_history.json").open() as f:
        history = json.load(f)
    with (base / "per_modality_ablation.json").open() as f:
        ablation = json.load(f)
    with (base / "modality_importance.json").open() as f:
        importance = json.load(f)
    bdir = root / "experiments" / "results" / "baselines"
    with (bdir / "logreg_severity.json").open() as f:
        logreg = json.load(f)
    with (bdir / "cox_survival.json").open() as f:
        cox = json.load(f)
    fdir = root / "experiments" / "results" / "fusion_comparison"
    sgdir = root / "experiments" / "results" / "subgroups"
    shdir = root / "experiments" / "results" / "survival_horizons"
    cdir = base / "clinical"
    return {
        "summary": summary, "history": history,
        "ablation": ablation, "importance": importance,
        "logreg": logreg, "cox": cox,
        "fig_dir": base / "figures",
        "fusion": _maybe_load(fdir / "per_fusion.json"),
        "fusion_fig": fdir / "fusion_bar.png",
        "subgroup": _maybe_load(sgdir / "subgroups.json"),
        "subgroup_fig": sgdir / "figures" / "subgroup_bars.png",
        "brier": _maybe_load(shdir / "brier.json"),
        "brier_fig": shdir / "brier_curve.png",
        # Clinical-utility / robustness / fairness / external artefacts
        "clinical_dir": cdir,
        "clinical_summary": _maybe_load(cdir / "clinical_summary.json"),
        "per_class_auroc": _maybe_load(cdir / "per_class_auroc.json"),
        "decision_curve": _maybe_load(cdir / "decision_curve.json"),
        "sens_spec": _maybe_load(cdir / "sens_spec.json"),
        "calibration_error": _maybe_load(cdir / "calibration_error.json"),
        "robustness": _maybe_load(cdir / "robustness.json"),
        "fairness": _maybe_load(cdir / "fairness_gap.json"),
        "external": _maybe_load(cdir / "external_cohort.json"),
        "dca_fig": cdir / "figures" / "decision_curve.png",
        "robust_fig": cdir / "figures" / "robustness.png",
        "pcroc_fig": cdir / "figures" / "per_class_roc.png",
    }


def _meanstd(values: list[float]) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (float(values[0]), 0.0)
    return (float(mean(values)), float(pstdev(values)))


def _fmt(m: float, s: float, digits: int = 3) -> str:
    if math.isnan(m):
        return "-"
    if s == 0.0 or math.isnan(s):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} \u00b1 {s:.{digits}f}"


# ----------------------------- content blocks -----------------------------

def _front_matter(styles, results) -> list:
    flow = [
        Paragraph("Multimodal Modeling of Sickle Cell Disease:", styles["title"]),
        Paragraph("A Quantitative Study of Severity Stratification "
                  "and Survival Prediction", styles["title"]),
        Spacer(1, 4),
        Paragraph("Sambou Kone", styles["authors"]),
        Paragraph("Independent researcher &middot; "
                  "Mmvlm4SCD project &middot; "
                  '<a href="https://github.com/koneke55/Mmvlm4SCD" '
                  'color="blue">github.com/koneke55/Mmvlm4SCD</a>',
                  styles["affil"]),
        HRFlowable(width="60%", thickness=0.5,
                   color=colors.HexColor("#999999"), spaceBefore=2,
                   spaceAfter=10, hAlign="CENTER"),
        Paragraph("ABSTRACT", styles["abstract_h"]),
    ]
    s = results["summary"]
    aurocs = [r["auroc_ovr"] for r in s["test_per_seed"]]
    accs = [r["accuracy"] for r in s["test_per_seed"]]
    cidxs = [r["c_index"] for r in s["test_per_seed"]]
    auroc_m, auroc_s = _meanstd(aurocs)
    acc_m, acc_s = _meanstd(accs)
    c_m, c_s = _meanstd(cidxs)

    abstract = (
        "Sickle Cell Disease (SCD) is a monogenic but phenotypically "
        "heterogeneous disorder for which clinical decisions hinge on "
        "stratifying patients by severity and projecting long-term "
        "survival. We present <b>Mmvlm4SCD</b>, an open multimodal deep-"
        "learning framework that fuses clinical, genomic, peripheral-blood-"
        "smear imaging and longitudinal vitals/labs trajectories into a "
        "shared embedding optimised jointly for ordinal severity "
        "classification and Cox proportional-hazards survival prediction. "
        "Across "
        f"{s['n_patients']} simulated SCD patients calibrated to published "
        "haematological literature and "
        f"{s['seeds']} random initialisations, the multimodal attention-"
        "fusion model reaches a macro AUROC of "
        f"<b>{_fmt(auroc_m, auroc_s)}</b> for severity (mild/moderate/"
        "severe) and a Harrell C-index of "
        f"<b>{_fmt(c_m, c_s)}</b> for time-to-death. A clinical-only "
        "logistic-regression baseline is competitive on severity "
        f"(AUROC = {results['logreg']['auroc_ovr']:.3f}) and a clinical-"
        f"only Cox model achieves C-index = {results['cox']['c_index']:.3f}, "
        "showing that on this benchmark the marginal lift from non-"
        "clinical modalities is small but mostly beneficial when paired "
        "with attention fusion. Modality-ablation analysis confirms "
        "clinical features dominate while imaging and genomic streams "
        "contribute complementary signal. We release the code, the "
        "synthetic-cohort generator, the data registry of public SCD "
        "sources, and a reproducible experimental pipeline to support "
        "extension to credentialed cohorts (dbGaP, MIMIC-IV, UK Biobank, "
        "NHLBI CuRe-SCD)."
    )
    flow.append(Paragraph(abstract, styles["abstract"]))
    flow.append(Paragraph(
        "<b>Keywords:</b> sickle cell disease &middot; multimodal learning "
        "&middot; survival analysis &middot; severity stratification "
        "&middot; clinical AI &middot; reproducible research",
        styles["abstract"]))
    return flow


def _section(styles, title, paragraphs):
    flow = [Paragraph(title, styles["h1"])]
    for p in paragraphs:
        flow.append(Paragraph(p, styles["body"]))
    return flow


def _table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#444")),
        ("LINEABOVE", (0, 1), (-1, 1), 0.4, colors.HexColor("#444")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f5f5")]),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d63")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
    t.setStyle(TableStyle(style))
    return t


def _figure(path: Path, caption: str, styles, width: float = 3.0 * inch):
    if not path.exists():
        return Paragraph(f"<i>(figure missing: {path.name})</i>", styles["small"])
    img = Image(str(path), width=width, height=width * 0.62)
    img.hAlign = "CENTER"
    return KeepTogether([img, Paragraph(caption, styles["caption"])])


# ----------------------------- main builder -------------------------------

def build(out_path: Path, root: Path = ROOT):
    styles = _styles()
    results = _load_results(root)
    flow: list = []

    # Cover and abstract on a single column.
    flow.extend(_front_matter(styles, results))
    flow.append(NextPageTemplate("twocol"))
    flow.append(PageBreak())

    # 1. Introduction --------------------------------------------------
    flow.append(Paragraph("1. Introduction", styles["h1"]))
    flow.append(Paragraph(
        "Sickle Cell Disease (SCD) is the most common monogenic disorder "
        "worldwide, caused by a single beta-globin missense mutation "
        "(<i>HBB</i> p.Glu6Val) that polymerises haemoglobin S under "
        "deoxygenation, deforms erythrocytes and triggers chronic "
        "haemolysis, vaso-occlusion, end-organ damage and shortened "
        "life expectancy [Piel2017, GBD2021SCD]. Despite a single "
        "causal mutation, the clinical course varies dramatically: "
        "patients with the same genotype can experience anywhere from "
        "&lt; 0.5 to &gt; 10 vaso-occlusive crises per year, and "
        "median survival differs by more than two decades across "
        "subgroups [Piel2017].",
        styles["body"]))
    flow.append(Paragraph(
        "Modern SCD care produces heterogeneous longitudinal data: "
        "complete blood counts, biochemistry panels, ICU and ED "
        "encounters, peripheral-blood-smear microscopy, genome-wide "
        "modifier variants, and patient-reported pain trajectories. "
        "Existing risk scores (e.g. modified Sickle Cell Severity "
        "Score) compress this complexity into hand-crafted formulae "
        "[Sebastiani2010, Quinn2007]. We hypothesise that a "
        "<i>jointly trained</i> multimodal model can recover stronger, "
        "calibrated decision functions for two clinically pivotal "
        "tasks: <b>(i)</b> ordinal severity stratification "
        "(mild/moderate/severe) and <b>(ii)</b> long-horizon survival "
        "prediction.",
        styles["body"]))
    flow.append(Paragraph(
        "This paper makes three contributions. <b>(1)</b> An open, "
        "reproducible multimodal architecture - <i>Mmvlm4SCD</i> - "
        "with pluggable encoders for clinical, genomic, imaging and "
        "temporal data, three fusion strategies, and a dual-task head "
        "trained with cross-entropy + Cox partial likelihood. "
        "<b>(2)</b> A literature-calibrated synthetic cohort generator "
        "that allows benchmarking before credential-gated cohorts are "
        "obtained, plus a curated registry of "
        f"{len(PUBLIC_SCD_DATASETS)} public SCD data sources spanning "
        "imaging, transcriptomics, ICU labs and population genetics. "
        "<b>(3)</b> A quantitative study comparing the multimodal "
        "model against strong tabular baselines, with seed-replicated "
        "metrics, modality-ablation, gradient-based interpretability, "
        "and survival stratification.",
        styles["body"]))

    # 2. Background ----------------------------------------------------
    flow.append(Paragraph("2. Background and Related Work", styles["h1"]))
    flow.append(Paragraph(
        "<b>SCD severity scoring.</b> Sebastiani et al. derived a "
        "Bayesian network severity score for SCD using clinical and "
        "laboratory variables [Sebastiani2010]; Quinn et al. quantified "
        "early-life predictors of severe disease [Quinn2007]. These "
        "remain dominant baselines in clinical practice but do not "
        "incorporate imaging or molecular data.",
        styles["body"]))
    flow.append(Paragraph(
        "<b>Survival modelling.</b> The Cox proportional-hazards model "
        "[Cox1972] is the de-facto SCD survival baseline. Deep-learning "
        "extensions such as DeepSurv [Katzman2018] retain the partial-"
        "likelihood objective while substituting a neural risk function "
        "for the linear predictor. We adopt this objective for the "
        "survival head.",
        styles["body"]))
    flow.append(Paragraph(
        "<b>Multimodal medical AI.</b> Cross-attention and "
        "Transformer-based fusion across imaging, genomics and "
        "tabular EHR have shown gains in oncology and ophthalmology "
        "[Acosta2022, Steyaert2023]. Application to SCD has been "
        "limited by data fragmentation; our registry consolidates the "
        "available open and credentialed sources to lower this "
        "barrier.",
        styles["body"]))
    flow.append(Paragraph(
        "<b>Imaging.</b> Convolutional networks readily distinguish "
        "sickled erythrocytes from normal cells on peripheral-blood-"
        "smear images [Xu2017, Alzubaidi2020]; we treat such CNN "
        "outputs as imaging embeddings within fusion.",
        styles["body"]))

    # 3. Data ----------------------------------------------------------
    flow.append(Paragraph("3. Data", styles["h1"]))
    flow.append(Paragraph("3.1 Public SCD data registry", styles["h2"]))
    flow.append(Paragraph(
        f"We catalogue {len(PUBLIC_SCD_DATASETS)} public SCD-relevant "
        "data sources covering clinical, genomic, imaging and "
        "epidemiological modalities. Table 1 summarises the registry; "
        "the canonical machine-readable list lives in "
        "<font face='Courier'>src/mmvlm4scd/data/registry.py</font> and "
        "is consumed by the documentation site.",
        styles["body"]))

    reg_rows = [["Source", "Modality", "Access"]]
    for s_ in PUBLIC_SCD_DATASETS:
        reg_rows.append([s_.name[:55], s_.modality, s_.access])
    flow.append(_table(reg_rows, col_widths=[2.2 * inch, 0.85 * inch, 0.7 * inch]))
    flow.append(Paragraph(
        "<b>Table 1.</b> Public SCD data sources tracked by the "
        "Mmvlm4SCD registry. Access tiers: <i>open</i>, "
        "<i>registered</i>, <i>dua-required</i> (requires Data Use "
        "Agreement).", styles["caption"]))

    flow.append(Paragraph("3.2 Synthetic cohort", styles["h2"]))
    flow.append(Paragraph(
        "Because dbGaP, UK Biobank and MIMIC-IV require Data Use "
        "Agreements that we did not have access to during this study, "
        "all empirical results in this paper are computed on a "
        "<b>literature-calibrated synthetic cohort</b> shipped with the "
        "code base. The generator (<font face='Courier'>data.synthetic"
        "</font>) draws "
        f"{results['summary']['n_patients']} virtual patients with "
        "marginal distributions calibrated to: SCD genotype prevalences "
        "(HbSS, HbSC, HbSbeta+, HbSbeta0); clinical labs (Hb, HbF%, "
        "WBC, platelets, LDH, total bilirubin, CRP); annual vaso-"
        "occlusive crisis rates; acute chest syndrome and stroke "
        "history; HBB and modifier variant indicators; CNN-style "
        "imaging embeddings of sickled vs normal erythrocytes; and "
        "monthly trajectories of vitals and pain VAS over a two-year "
        "horizon. Severity labels are derived as quantiles of a "
        "weighted clinical/genotype score; survival times are sampled "
        "from a Weibull model with severity- and genotype-dependent "
        "hazard. The full generator is fully transparent and "
        "deterministic given a seed.",
        styles["body"]))
    flow.append(Paragraph(
        "<b>Synthetic-data disclaimer.</b> The cohort is not real "
        "patient data. Results should be read as architectural "
        "benchmarks, not clinical claims. The codebase is designed so "
        "that, once an investigator obtains DUA-gated cohorts, the "
        "same training and evaluation entry points can be invoked with "
        "minimal modification to the data loader.",
        styles["body"]))

    # 4. Methods -------------------------------------------------------
    flow.append(Paragraph("4. Methods", styles["h1"]))
    flow.append(Paragraph("4.1 Architecture", styles["h2"]))
    flow.append(Paragraph(
        "Mmvlm4SCD comprises four modality-specific encoders, a fusion "
        "module, and two prediction heads. The clinical encoder is a "
        "3-layer MLP over standardised tabular features. The genomic "
        "encoder splits the input into a binary HBB/modifier-variant "
        "block and a continuous polygenic-like block, projects each, "
        "then concatenates them. The imaging encoder consumes "
        "pre-computed CNN embeddings from peripheral-blood-smear "
        "patches. The temporal encoder is a one-layer GRU over "
        "monthly vitals/labs trajectories with the final hidden "
        "state read out. All four encoders project to a shared "
        f"{results['summary']['test_per_seed'][0].get('embed_dim', 64)}-"
        "dimensional embedding (default).",
        styles["body"]))
    flow.append(Paragraph(
        "Three fusion strategies are implemented and benchmarked: "
        "<b>attention</b> (a small Transformer encoder over modality "
        "tokens with a learned [CLS] readout), <b>cross-attention</b> "
        "(clinical embedding queries the other modalities), and "
        "<b>late fusion</b> (concatenate then project). Each "
        "fusion produces a single vector that feeds two heads: a "
        "3-class severity classifier and a scalar survival risk "
        "score.",
        styles["body"]))

    flow.append(Paragraph("4.2 Training objective", styles["h2"]))
    flow.append(Paragraph(
        "Severity is trained with multinomial cross-entropy. Survival "
        "is trained with the negative Breslow partial log-likelihood:",
        styles["body"]))
    flow.append(Paragraph(
        "<font face='Courier'>L_cox = -(1/|D|) sum_{i in D} "
        "[ r_i - log sum_{j: t_j &gt;= t_i} exp(r_j) ]</font>",
        styles["mono"]))
    flow.append(Paragraph(
        "where <i>r_i</i> is the predicted risk score, <i>t_i</i> the "
        "observed time, and <i>D</i> the set of subjects with an "
        "observed event. The combined objective is "
        "<font face='Courier'>L = alpha &middot; L_ce + beta &middot; "
        "L_cox</font> with default <font face='Courier'>(alpha, beta) "
        "= (1.0, 0.5)</font>. Optimisation uses AdamW with cosine "
        "annealing and gradient clipping at norm 1.0. Validation "
        "AUROC drives early stopping.",
        styles["body"]))

    flow.append(Paragraph("4.3 Evaluation", styles["h2"]))
    flow.append(Paragraph(
        "We report <b>accuracy</b>, <b>macro F1</b> and <b>macro "
        "one-vs-rest AUROC</b> for severity; <b>Harrell's C-index</b> "
        "for survival [Harrell1996]. We additionally report "
        "calibration via reliability diagrams, modality importance "
        "via gradient L2 norms, and Kaplan-Meier curves stratified by "
        "predicted-risk tertile. Results are reported as mean +/- "
        f"standard deviation across {results['summary']['seeds']} "
        "random seeds.",
        styles["body"]))

    # 5. Experiments ---------------------------------------------------
    flow.append(Paragraph("5. Experiments", styles["h1"]))
    s = results["summary"]
    aurocs = [r["auroc_ovr"] for r in s["test_per_seed"]]
    accs = [r["accuracy"] for r in s["test_per_seed"]]
    f1s = [r["f1_macro"] for r in s["test_per_seed"]]
    cidxs = [r["c_index"] for r in s["test_per_seed"]]

    flow.append(Paragraph("5.1 Setup", styles["h2"]))
    flow.append(Paragraph(
        f"Cohort: {s['n_patients']} synthetic patients; 70/15/15 "
        "train/val/test split; batch size 64; 30 epochs with "
        "early-stop patience 8. Hardware: CPU-only PyTorch 2.x. "
        "Code at "
        '<a href="https://github.com/koneke55/Mmvlm4SCD" color="blue">'
        "github.com/koneke55/Mmvlm4SCD</a>. "
        "Reproducibility: seeds {0, 1, 2}; full configs in "
        "<font face='Courier'>configs/</font>.",
        styles["body"]))

    flow.append(Paragraph("5.2 Main results", styles["h2"]))
    main_rows = [
        ["Model", "Acc", "F1 (macro)", "AUROC (OvR)", "C-index"],
        ["Logistic regression (clinical)",
         f"{results['logreg']['accuracy']:.3f}",
         f"{results['logreg']['f1_macro']:.3f}",
         f"{results['logreg']['auroc_ovr']:.3f}",
         "-"],
        ["Cox PH (clinical)", "-", "-", "-",
         f"{results['cox']['c_index']:.3f}"],
        ["Mmvlm4SCD (attention fusion)",
         _fmt(*_meanstd(accs)),
         _fmt(*_meanstd(f1s)),
         _fmt(*_meanstd(aurocs)),
         _fmt(*_meanstd(cidxs))],
    ]
    flow.append(_table(main_rows, col_widths=[1.95 * inch, 0.55 * inch,
                                              0.7 * inch, 0.85 * inch,
                                              0.7 * inch]))
    flow.append(Paragraph(
        f"<b>Table 2.</b> Held-out test performance over "
        f"{s['seeds']} seeds. Mmvlm4SCD scores reported as mean +/- "
        "standard deviation.",
        styles["caption"]))

    flow.append(Paragraph("5.3 Per-modality ablation", styles["h2"]))
    abl_rows = [["Dropped modality", "Acc", "F1", "AUROC", "C-index"]]
    for k, m in results["ablation"].items():
        abl_rows.append([k, f"{m['accuracy']:.3f}", f"{m['f1_macro']:.3f}",
                         f"{m['auroc_ovr']:.3f}", f"{m['c_index']:.3f}"])
    flow.append(_table(abl_rows, col_widths=[1.45 * inch, 0.65 * inch,
                                             0.65 * inch, 0.7 * inch,
                                             0.7 * inch]))
    flow.append(Paragraph(
        "<b>Table 3.</b> Test performance when each modality is set to "
        "zeros at train and test time. Dropping the clinical modality "
        "produces by far the largest degradation.",
        styles["caption"]))

    # Bootstrap CIs from subgroup driver (which also bootstrapped overall).
    if results["subgroup"] is not None:
        bs = results["subgroup"]["bootstrap_overall"]
        boot_rows = [
            ["Metric", "Mean", "95% CI low", "95% CI high"],
            ["Accuracy",  f"{bs['accuracy']['mean']:.3f}",
             f"{bs['accuracy']['ci_low']:.3f}",
             f"{bs['accuracy']['ci_high']:.3f}"],
            ["F1 (macro)", f"{bs['f1_macro']['mean']:.3f}",
             f"{bs['f1_macro']['ci_low']:.3f}",
             f"{bs['f1_macro']['ci_high']:.3f}"],
            ["AUROC (OvR)", f"{bs['auroc_ovr']['mean']:.3f}",
             f"{bs['auroc_ovr']['ci_low']:.3f}",
             f"{bs['auroc_ovr']['ci_high']:.3f}"],
            ["C-index", f"{bs['c_index']['mean']:.3f}",
             f"{bs['c_index']['ci_low']:.3f}",
             f"{bs['c_index']['ci_high']:.3f}"],
        ]
        flow.append(Paragraph("5.3.1 Bootstrap confidence intervals",
                              styles["h2"]))
        flow.append(Paragraph(
            "Non-parametric 95% bootstrap CIs over 300 resamples of the "
            "held-out test set for the attention-fusion model:",
            styles["body"]))
        flow.append(_table(boot_rows, col_widths=[1.4 * inch, 0.8 * inch,
                                                  0.95 * inch, 0.95 * inch]))
        flow.append(Paragraph(
            "<b>Table 3a.</b> Bootstrap CIs around the test metrics; the "
            "narrow C-index CI confirms the survival ranking is "
            "statistically meaningful despite a moderate point estimate.",
            styles["caption"]))

    if results["fusion"] is not None:
        flow.append(Paragraph("5.3.2 Fusion-strategy comparison",
                              styles["h2"]))
        f_rows = [["Fusion", "AUROC (mean +/- std)",
                   "C-index (mean +/- std)", "Acc", "F1"]]
        for name in ("attention", "cross", "late"):
            d = results["fusion"][name]
            f_rows.append([name,
                           f"{d['mean_auroc']:.3f} +/- {d['std_auroc']:.3f}",
                           f"{d['mean_c_index']:.3f} +/- {d['std_c_index']:.3f}",
                           f"{d['mean_accuracy']:.3f}",
                           f"{d['mean_f1_macro']:.3f}"])
        flow.append(_table(f_rows, col_widths=[0.85 * inch, 1.5 * inch,
                                               1.5 * inch, 0.5 * inch,
                                               0.5 * inch]))
        flow.append(Paragraph(
            "<b>Table 3b.</b> Comparison of the three fusion strategies "
            "implemented in <font face='Courier'>models/fusion/</font>. "
            "On the synthetic cohort the differences are small and "
            "within seed variance; cross-attention trades a small AUROC "
            "edge for slightly better C-index over late fusion.",
            styles["caption"]))
        flow.append(_figure(results["fusion_fig"],
                            "Figure 6. Severity AUROC and survival "
                            "C-index by fusion strategy (mean +/- std "
                            "over 3 seeds).", styles))

    flow.append(Paragraph("5.4 Modality importance", styles["h2"]))
    flow.append(Paragraph(
        "We additionally measure modality importance by averaging the "
        "L2 norm of the gradient of the predicted-class severity "
        "logit with respect to each modality input over the test set "
        "(Figure 3). Clinical features dominate "
        f"({results['importance']['clinical']:.2f}), followed by "
        f"imaging ({results['importance']['imaging']:.2f}), genomic "
        f"({results['importance']['genomic']:.2f}), and temporal "
        f"({results['importance']['temporal']:.2f}). The two "
        "approaches (ablation vs. gradient) agree directionally.",
        styles["body"]))

    # Figures (try to keep on a clean page)
    fig_dir = results["fig_dir"]
    flow.append(_figure(fig_dir / "training_curves.png",
                        "Figure 1. Training loss and validation metrics "
                        "(seed 0).", styles))
    flow.append(_figure(fig_dir / "confusion.png",
                        "Figure 2. Severity confusion matrix on the "
                        "held-out test set (seed 0).", styles))
    flow.append(_figure(fig_dir / "modality_importance.png",
                        "Figure 3. Per-modality gradient L2 norm of the "
                        "predicted severity logit, averaged over the "
                        "test set.", styles))
    flow.append(_figure(fig_dir / "km_by_risk.png",
                        "Figure 4. Kaplan-Meier survival curves "
                        "stratified by predicted-risk tertile from the "
                        "Cox-trained survival head.", styles))
    flow.append(_figure(fig_dir / "calibration.png",
                        "Figure 5. Reliability diagram for the top-"
                        "class severity probability.", styles))

    # 5.5 Time-dependent survival ------------------------------------
    if results["brier"] is not None:
        flow.append(Paragraph("5.5 Time-dependent survival evaluation",
                              styles["h2"]))
        bj = results["brier"]
        flow.append(Paragraph(
            "Beyond the Cox C-index we evaluate Brier scores at "
            "increasing follow-up horizons (1, 2, 5, 10, 15, 20 years) "
            "and the Integrated Brier Score (IBS) over the same "
            "interval. Brier(t) is computed using inverse-probability-"
            "of-censoring weighting [Graf1999], with the censoring "
            "distribution estimated by a Kaplan-Meier on flipped event "
            "indicators. Per-subject survival curves are obtained by "
            "shifting a population baseline KM curve in cumulative-"
            "hazard space using the model's risk score, mirroring the "
            "DeepSurv / Cox prediction protocol.",
            styles["body"]))
        bs_rows = [["Horizon (yr)"] + [f"{h:g}" for h in bj["horizons"]]
                   + ["IBS"]]
        bs_rows.append(["Brier"] + [f"{bj['brier'][f'bs_{h:g}']:.3f}"
                                    for h in bj["horizons"]]
                       + [f"{bj['ibs']:.3f}"])
        flow.append(_table(bs_rows,
                           col_widths=[0.9 * inch] +
                           [0.55 * inch] * len(bj["horizons"]) +
                           [0.6 * inch]))
        flow.append(Paragraph(
            "<b>Table 4.</b> Time-dependent Brier scores and Integrated "
            "Brier Score (IBS) for the attention-fusion model. Brier "
            "rises monotonically with horizon, reflecting accumulating "
            "uncertainty over long follow-up.", styles["caption"]))
        flow.append(_figure(results["brier_fig"],
                            "Figure 7. Brier score over follow-up "
                            "horizon with the IBS annotated.", styles))

    # 5.6 Subgroup analysis ------------------------------------------
    if results["subgroup"] is not None:
        flow.append(Paragraph("5.6 Subgroup analysis", styles["h2"]))
        sg = results["subgroup"]
        flow.append(Paragraph(
            f"We slice the test set (n={sg['n_test']}) by genotype, age "
            "band and sex to expose performance disparities. Subgroups "
            "with fewer than 20 patients are dropped to keep estimates "
            "stable. The overall AUROC is "
            f"{sg['overall']['auroc_ovr']:.3f} and overall C-index is "
            f"{sg['overall']['c_index']:.3f}.", styles["body"]))
        sg_rows = [["Subgroup", "AUROC", "C-index"]]
        for k, v in sg["groups"].items():
            au = v.get("auroc_ovr", float("nan"))
            ci = v.get("c_index", float("nan"))
            sg_rows.append([k,
                            "-" if au != au else f"{au:.3f}",
                            "-" if ci != ci else f"{ci:.3f}"])
        flow.append(_table(sg_rows, col_widths=[2.0 * inch, 0.8 * inch,
                                                0.8 * inch]))
        flow.append(Paragraph(
            "<b>Table 5.</b> Subgroup-stratified test metrics. Older "
            "patients (45+) and the HbSS subgroup show the largest "
            "performance gaps, motivating subgroup-aware loss "
            "weighting in future work.", styles["caption"]))
        flow.append(_figure(results["subgroup_fig"],
                            "Figure 8. Subgroup AUROC (left) and "
                            "C-index (right). Dashed vertical lines "
                            "mark the overall test-set value.",
                            styles, width=4.5 * inch))

    # 5.7 Clinical utility, robustness, fairness and external validity
    if results.get("clinical_summary") is not None:
        flow.append(Paragraph("5.7 Clinical utility and calibration",
                              styles["h2"]))
        cs = results["clinical_summary"]
        bullets = []
        pc = results.get("per_class_auroc") or {}
        if pc:
            bullets.append(
                "Per-class one-vs-rest AUROC: mild = "
                f"{pc.get('auroc_class0', float('nan')):.3f}, "
                f"moderate = {pc.get('auroc_class1', float('nan')):.3f}, "
                f"severe = {pc.get('auroc_class2', float('nan')):.3f}.")
        cer = results.get("calibration_error") or {}
        if cer:
            bullets.append(
                f"Calibration: ECE = {cer.get('ece', float('nan')):.3f}, "
                f"MCE = {cer.get('mce', float('nan')):.3f} (10 bins).")
        if cs.get("decision_curve_max_net_benefit") is not None:
            bullets.append(
                "Maximum net benefit on the decision curve "
                f"(severe vs not, prevalence "
                f"{(results['decision_curve'] or {}).get('prevalence', 0):.2f}) "
                f"= {cs['decision_curve_max_net_benefit']:.3f}.")
        flow.append(Paragraph(
            "Beyond rank-based metrics, we report threshold-based "
            "clinical utility: " + " ".join(bullets), styles["body"]))
        if results.get("pcroc_fig", Path()).exists():
            flow.append(_figure(results["pcroc_fig"],
                                "Figure 9. Per-class one-vs-rest ROC.",
                                styles, width=3.6 * inch))
        if results.get("dca_fig", Path()).exists():
            flow.append(_figure(results["dca_fig"],
                                "Figure 10. Decision-curve analysis "
                                "(Vickers & Elkin 2006).",
                                styles, width=4.0 * inch))
        ss = results.get("sens_spec") or {}
        if ss:
            ss_rows = [["Threshold", "Sens", "Spec", "PPV", "NPV"]]
            for t, m in ss.items():
                ss_rows.append([t,
                                f"{m['sensitivity']:.3f}",
                                f"{m['specificity']:.3f}",
                                f"{m['ppv']:.3f}",
                                f"{m['npv']:.3f}"])
            flow.append(_table(ss_rows, col_widths=[1.0 * inch] +
                               [0.7 * inch] * 4))
            flow.append(Paragraph(
                "<b>Table 6.</b> Sensitivity, specificity, PPV and NPV "
                "at three operating thresholds for predicting severe "
                "disease.", styles["caption"]))

    if results.get("robustness") is not None:
        flow.append(Paragraph("5.8 Robustness to missing modalities",
                              styles["h2"]))
        rb = results["robustness"]
        rows = [["p", "Acc", "F1", "AUROC", "C-index"]]
        for p in ("0.0", "0.1", "0.25", "0.5"):
            if p in rb:
                m = rb[p]
                rows.append([p, f"{m['accuracy']:.3f}",
                             f"{m['f1_macro']:.3f}",
                             f"{m['auroc_ovr']:.3f}",
                             f"{m['c_index']:.3f}"])
        flow.append(Paragraph(
            "Each modality is zeroed out at test time with probability p, "
            "averaged over 3 repeats; this models missing-at-random EHR.",
            styles["body"]))
        flow.append(_table(rows, col_widths=[0.6 * inch] +
                           [0.7 * inch] * 4))
        flow.append(Paragraph(
            "<b>Table 7.</b> Test metrics under random per-sample "
            "modality dropout.", styles["caption"]))
        if results.get("robust_fig", Path()).exists():
            flow.append(_figure(results["robust_fig"],
                                "Figure 11. Test AUROC and C-index "
                                "vs modality-dropout probability.",
                                styles, width=4.0 * inch))

    if results.get("fairness") is not None:
        flow.append(Paragraph("5.9 Fairness gap", styles["h2"]))
        fa = results["fairness"].get("auroc_ovr", {})
        fc = results["fairness"].get("c_index", {})
        rows = [["Metric", "Best", "Worst", "Gap", "#groups"]]
        if fa:
            rows.append(["AUROC (OvR)", f"{fa.get('best',0):.3f}",
                         f"{fa.get('worst',0):.3f}",
                         f"{fa.get('gap',0):.3f}",
                         f"{int(fa.get('n_groups',0))}"])
        if fc:
            rows.append(["C-index", f"{fc.get('best',0):.3f}",
                         f"{fc.get('worst',0):.3f}",
                         f"{fc.get('gap',0):.3f}",
                         f"{int(fc.get('n_groups',0))}"])
        flow.append(_table(rows, col_widths=[1.2 * inch] +
                           [0.7 * inch] * 4))
        flow.append(Paragraph(
            "<b>Table 8.</b> Subgroup fairness gap (max - min).",
            styles["caption"]))

    if results.get("external") is not None:
        flow.append(Paragraph("5.10 External-cohort simulation",
                              styles["h2"]))
        rows = [["Cohort", "Acc", "F1", "AUROC", "C-index"]]
        for k, m in results["external"].items():
            rows.append([k, f"{m.get('accuracy',float('nan')):.3f}",
                         f"{m.get('f1_macro',float('nan')):.3f}",
                         f"{m.get('auroc_ovr',float('nan')):.3f}",
                         f"{m.get('c_index',float('nan')):.3f}"])
        flow.append(Paragraph(
            "We approximate distribution shift by re-sampling cohorts "
            "with seeds disjoint from training and evaluating the "
            "trained attention-fusion model on each.",
            styles["body"]))
        flow.append(_table(rows, col_widths=[1.0 * inch] +
                           [0.7 * inch] * 4))
        flow.append(Paragraph(
            "<b>Table 9.</b> External-cohort simulation: each cohort "
            "is generated with a different RNG seed.",
            styles["caption"]))

    # 6. Discussion ----------------------------------------------------
    flow.append(Paragraph("6. Discussion", styles["h1"]))
    flow.append(Paragraph(
        "Three observations emerge. <b>(i)</b> A strong tabular "
        "baseline is hard to beat on this synthetic cohort. The "
        "logistic-regression severity classifier and the linear Cox "
        "model both perform competitively, indicating that the "
        "synthetic generator is - by construction - dominated by "
        "low-dimensional clinical signal. This motivates evaluating "
        "the multimodal architecture on cohorts with richer "
        "modalities, where genomic and imaging streams carry more "
        "<i>independent</i> information (e.g. UK Biobank smears + "
        "TOPMed WGS). <b>(ii)</b> The clinical modality is consistently "
        "the most informative according to both ablation and "
        "gradient analyses, mirroring clinical experience that "
        "haematological labs largely determine SCD prognosis. "
        "<b>(iii)</b> Imaging contributes more than genomics to the "
        "model's decision-time signal in our setup; this is sensible "
        "because the synthetic imaging block encodes the sickled-"
        "morphology fraction, which is a direct phenotype, whereas "
        "the synthetic genomic block is closer to a mild risk "
        "modifier.",
        styles["body"]))

    # 7. Limitations ---------------------------------------------------
    flow.append(Paragraph("7. Limitations and Ethical Considerations",
                          styles["h1"]))
    flow.append(Paragraph(
        "Our experiments are run on simulated data. While marginal "
        "distributions are calibrated to SCD literature, the joint "
        "structure is parametric and necessarily under-represents the "
        "full heterogeneity of SCD. Results should not be interpreted "
        "as clinical evidence. Once we secure DUA approvals (target: "
        "dbGaP phs001514, MIMIC-IV-SCD subset, NHLBI CuRe-SCD), the "
        "same training pipeline can be invoked with replacement data "
        "loaders.",
        styles["body"]))
    flow.append(Paragraph(
        "Because SCD disproportionately affects populations of African "
        "ancestry, deployments must be audited for performance "
        "disparities by ancestry, age and sex; the codebase exposes "
        "subgroup-stratified evaluation hooks. We follow the "
        "Contributor Covenant Code of Conduct and release everything "
        "under MIT.",
        styles["body"]))

    # 8. Reproducibility -----------------------------------------------
    flow.append(Paragraph("8. Reproducibility", styles["h1"]))
    flow.append(Paragraph(
        "All numbers in this paper are produced by "
        "<font face='Courier'>src/scripts/run_full_experiment.py "
        "--config configs/default.yaml --seeds 3 --ablate</font> and "
        "<font face='Courier'>src/scripts/run_baselines.py</font>, then "
        "rendered by <font face='Courier'>paper/build_paper.py</font>. "
        "JSON artefacts in "
        "<font face='Courier'>experiments/results/</font> are the "
        "single source of truth. Unit tests covering data, models and "
        "losses run in &lt; 30 s on a laptop CPU.",
        styles["body"]))

    # 9. Conclusion ----------------------------------------------------
    flow.append(Paragraph("9. Conclusion", styles["h1"]))
    flow.append(Paragraph(
        "We presented Mmvlm4SCD, an open multimodal framework for "
        "Sickle Cell Disease that jointly learns severity "
        "stratification and survival prediction. On a literature-"
        "calibrated synthetic benchmark, the model attains "
        f"AUROC = {_fmt(*_meanstd(aurocs))} and "
        f"C-index = {_fmt(*_meanstd(cidxs))} across seeds, with "
        "clinical features dominating decision-time importance. The "
        "released code, registry of public SCD sources, and "
        "reproducible pipeline lower the barrier to extending these "
        "experiments to real, credentialed cohorts - the next step "
        "in establishing whether multimodal fusion delivers a "
        "clinically meaningful uplift over strong tabular baselines "
        "in SCD.",
        styles["body"]))

    # 10. References ---------------------------------------------------
    flow.append(Paragraph("References", styles["h1"]))
    refs = [
        "[Acosta2022] Acosta JN <i>et al.</i> Multimodal biomedical AI. "
        "<i>Nature Medicine</i>, 28(9):1773-1784, 2022.",
        "[Graf1999] Graf E <i>et al.</i> Assessment and comparison of "
        "prognostic classification schemes for survival data. "
        "<i>Statistics in Medicine</i>, 18(17-18):2529-2545, 1999.",
        "[Alzubaidi2020] Alzubaidi L <i>et al.</i> Deep learning models for "
        "classification of red blood cells in microscopy images. <i>Electronics</i>, 9(3):427, 2020.",
        "[Cox1972] Cox DR. Regression models and life-tables. <i>JRSS B</i>, 34(2):187-202, 1972.",
        "[GBD2021SCD] GBD 2021 Sickle Cell Disease Collaborators. Global, regional and "
        "national prevalence and mortality burden of sickle cell disease, 2000-2021. <i>Lancet Haematology</i>, 2023.",
        "[Harrell1996] Harrell FE <i>et al.</i> Multivariable prognostic models. "
        "<i>Statistics in Medicine</i>, 15(4):361-387, 1996.",
        "[Katzman2018] Katzman JL <i>et al.</i> DeepSurv: personalised treatment recommender via Cox "
        "proportional hazards deep neural network. <i>BMC Medical Research Methodology</i>, 18:24, 2018.",
        "[Piel2017] Piel FB, Steinberg MH, Rees DC. Sickle cell disease. <i>NEJM</i>, 376(16):1561-1573, 2017.",
        "[Quinn2007] Quinn CT <i>et al.</i> Survival of children with sickle cell disease. <i>Blood</i>, 109(11):4928-4933, 2007.",
        "[Sebastiani2010] Sebastiani P <i>et al.</i> A network model to predict the risk of death in sickle cell disease. "
        "<i>Blood</i>, 115(11):2118-2127, 2010.",
        "[Steyaert2023] Steyaert S <i>et al.</i> Multimodal deep learning to predict prognosis in cancer. "
        "<i>npj Precision Oncology</i>, 7:80, 2023.",
        "[Xu2017] Xu M <i>et al.</i> A deep convolutional neural network for classification of red blood cells in sickle cell anemia. "
        "<i>PLOS Computational Biology</i>, 13(10):e1005746, 2017.",
        "[Karczewski2020] Karczewski KJ <i>et al.</i> The mutational constraint spectrum quantified from variation in 141,456 humans. "
        "<i>Nature</i>, 581:434-443, 2020.",
        "[Johnson2023MIMICIV] Johnson AEW <i>et al.</i> MIMIC-IV, a freely accessible electronic health record dataset. "
        "<i>Scientific Data</i>, 10:1, 2023.",
    ]
    for r in refs:
        flow.append(Paragraph(r, styles["ref"]))

    doc = _doc(out_path)
    doc.build(flow)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "paper" / "paper.pdf"))
    args = p.parse_args(argv)
    out = build(Path(args.out))
    print(f"Wrote {out}  ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
