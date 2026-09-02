from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECK_ROOT = PROJECT_ROOT / "docs" / "meeting" / "tangent_subspace_connectivity"
QMD_PATH = DECK_ROOT / "tangent_subspace_connectivity.qmd"
HTML_PATH = DECK_ROOT / "tangent_subspace_connectivity.html"
PLAN_PATH = (
    PROJECT_ROOT / "docs" / "meeting" / ("tangent_subspace_connectivity_slide_plan.md")
)
KOREAN_PATTERN = re.compile(r"[\uac00-\ud7a3]")


def _slides(qmd: str) -> list[str]:
    return re.split(r"(?m)^## ", qmd)[1:]


def _visible_text(slide: str) -> str:
    return slide.split("::: {.notes}", maxsplit=1)[0]


def _notes(slide: str) -> str:
    return slide.split("::: {.notes}", maxsplit=1)[1].rsplit(":::", maxsplit=1)[0]


def test_deck_has_exact_47_slide_contract() -> None:
    qmd = QMD_PATH.read_text()
    slides = _slides(qmd)

    assert len(slides) == 47
    assert qmd.count("::: {.notes}") == 47
    assert qmd.count("::: {.notes}") == qmd.count("<!-- speaker-note-end -->")
    assert "From Exact Balance to Geometry-Aware Response Alignment" in slides[0]
    assert "Pentagram Benchmark" in slides[32]
    assert "Pentagram q50 Test Source" in slides[33]
    assert "Pentagram Test-Set Distribution" in slides[35]
    assert "Pentagram Accuracy Improves" in slides[36]
    assert "Unit-Square Benchmark" in slides[40]
    assert "Unit-Square q50 Test Source" in slides[41]
    assert "Unit-Square Test Errors" in slides[43]
    assert "Unit-Square Source-Count Study" in slides[44]
    assert "A Common Source Budget" in slides[45]
    assert "Three Levels of Evidence" in slides[46]


def test_visible_text_is_english_and_notes_are_korean() -> None:
    for slide in _slides(QMD_PATH.read_text()):
        assert not KOREAN_PATTERN.search(_visible_text(slide))
        assert KOREAN_PATTERN.search(_notes(slide))


def test_formula_and_semantic_contract_is_code_consistent() -> None:
    qmd = QMD_PATH.read_text()

    required = (
        r"\widetilde\phi=\frac12\left[f+(p-q)\right]",
        r"\widetilde\psi=\frac12\left[f-(p-q)\right]",
        r"S=H_x+H_y",
        r"m(\delta)=m_0+S\delta",
        r"g(\delta)=S^\top M_\Omega\left(m_0+S\delta\right)",
        r"A=S^\top M_\Omega S",
        r"Az=S^\top M_\Omega(Sz)",
        r"z_0=D^{-1}g_0",
        r"v_0=Sz_0",
        r"\delta_K=-\sum_{k=0}^{K-1}c_kz_k",
        r"\phi_K=\widetilde\phi+\delta_K",
        r"\psi_K=\widetilde\psi-\delta_K",
        r"d_A(i,j)=\left\lceil\frac{d_L(i,j)}{2}\right\rceil",
    )
    for formula in required:
        assert formula in qmd

    assert "not the exact diagonal of" in qmd
    assert "Krylov-like nested response subspace" in qmd
    assert "The production gradient is generally dense" in qmd
    assert "localized canonical probe" in qmd
    assert "does not prove a PDE-specific optimal K" in qmd
    assert "No global matrix is assembled" in qmd


def test_frozen_numerical_claims_and_evidence_badges_are_present() -> None:
    qmd = QMD_PATH.read_text()

    for badge in (
        "EXACT ALGEBRA",
        "PRODUCTION ALGORITHM",
        "STRUCTURAL PROXY",
        "EMPIRICAL RESULT",
    ):
        assert badge in qmd
    for claim in (
        "31.2106%",
        "98.2666%",
        "99.8688%",
        "2.678%",
        "1.590%",
        "1.234%",
        "1.112%",
        "141.373",
        "361.773",
        "0.4255%",
        "0.3505%",
        "4,800",
        "sample 79",
        "sample 11",
        "3.238%",
        "0.819%",
    ):
        assert claim in qmd


def test_problem_field_and_distribution_contracts_are_present() -> None:
    qmd = QMD_PATH.read_text()

    for formula_or_claim in (
        r"-\nabla\!\cdot\!\left(a\nabla u\right)",
        r"\mathbf b=\left(-\frac{y}{2R},\frac{x}{2R}\right)",
        r"-\Delta u=f",
        r"a(x,y)=1",
        r"\widehat\phi+\widehat\psi=f",
        "Reference $\\phi^\\star,\\psi^\\star$ are evaluation-only.",
        "100 samples | K=4 best-energy checkpoint",
        "100 samples | 4,800-source seed-0 best-energy checkpoint",
    ):
        assert formula_or_claim in qmd

    assert qmd.count("phi와 psi의 개별 error distribution") == 2
    assert "assets/pentagram_problem_coefficients.html" in qmd
    assert "assets/pentagram_sample79_directional.html" in qmd
    assert "assets/pentagram_sample79_solution.html" in qmd
    assert "assets/pentagram_test_distribution.html" in qmd
    assert "assets/unit_square_problem_coefficient.html" in qmd
    assert "assets/unit_square_sample11_directional.html" in qmd
    assert "assets/unit_square_sample11_solution.html" in qmd
    assert "assets/unit_square_test_distribution.html" in qmd


def test_fragments_have_matching_korean_click_cues() -> None:
    for slide in _slides(QMD_PATH.read_text()):
        indices = {
            int(value)
            for value in re.findall(
                r'data-fragment-index="(\d+)"', _visible_text(slide)
            )
        }
        notes = _notes(slide)
        assert notes.count("**Click") == len(indices)
        if indices:
            assert indices == set(range(max(indices) + 1))


def test_qmd_references_only_deck_local_assets() -> None:
    qmd = QMD_PATH.read_text()
    iframe_sources = re.findall(r'<iframe[^>]+src="([^"]+)"', qmd)

    assert iframe_sources
    assert all(source.startswith("assets/") for source in iframe_sources)
    assert "http://" not in qmd
    assert "https://" not in qmd


def test_rendered_html_is_offline_and_has_exact_slide_count() -> None:
    html = HTML_PATH.read_text()

    assert len(re.findall(r'<section[^>]+class="[^"]*slide level2[^"]*"', html)) == 47
    assert "cdn.plot.ly" not in html
    assert re.search(r'(?:src|href)="https?://', html) is None
    assert "tangent_subspace_connectivity_files/libs/revealjs/dist/reveal.js" in html


def test_slide_plan_and_qa_contract_exist() -> None:
    plan = PLAN_PATH.read_text()
    qa_source = (DECK_ROOT / "qa_reveal.js").read_text()

    assert "47 main slides" in plan
    assert "Visible language: English" in plan
    assert "Speaker notes: Korean" in plan
    assert "1600x900" in qa_source
    assert "1280x720" in qa_source
    assert "expectedSlideCount = 47" in qa_source
