from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECK_ROOT = PROJECT_ROOT / "docs" / "meeting" / "annulus_transition_error"
QMD_PATH = DECK_ROOT / "annulus_transition_error.qmd"
HTML_PATH = DECK_ROOT / "annulus_transition_error.html"
STYLES_PATH = DECK_ROOT / "styles.scss"
SLIDE_PLAN_PATH = (
    PROJECT_ROOT / "docs" / "meeting" / ("annulus_transition_error_slide_plan.md")
)
QA_REPORT_PATH = DECK_ROOT / "screenshots" / "qa" / "qa_report.json"

EXPECTED_TITLES = [
    "Transition Structure Appears in Directional Sources and Reconstructions",
    "Where Line Length Enters Projection and Pull-Back",
    "Locating the Transition Error in the Reconstruction Pipeline",
    "Projection Strategies Tested for the Annulus Transition",
    "Three Ways to Choose the Cross-Axis Reconstruction Weight",
    "Geometry-Only Compact C2: Encode Known Topology",
    "Mismatch-Detected Seam C2: Detect, Then Blend",
    "Local Weak-Residual Reliability: Trust the Better PDE Candidate",
    "Poisson: Local PDE Reliability Gives the Strongest Post-Hoc Improvement",
    "CDR: The Same Estimator Ordering Persists",
    "Pure Poisson: Directional Sources and Weak-Residual Reconstruction",
    "Pure Poisson: Signed Errors and Test-Set Accuracy",
    "CDR: Physical Coefficients, Directional Sources, and Reconstruction",
    "CDR: Signed Errors and Test-Set Accuracy",
    "Exact Balance Does Not Imply Directional Response Consistency",
    "Next Candidate: Tangent Correction Within the Balance Plane",
    "Backup A: Unit and Physical Green Integrals Are Equivalent",
    "Backup B: Exact Green Error Decomposition",
]
EXPECTED_FRAGMENT_COUNTS = [
    1,
    4,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    2,
    3,
    2,
    2,
    3,
    0,
    0,
]
EXPECTED_ASSETS = [
    "annulus_transition_sample47_error_matrix.html",
    "annulus_transition_sample47_error_matrix_marked.html",
    "geometry_c2_method_sample0.html",
    "mismatch_seam_c2_method_sample0.html",
    "weak_residual_reliability_method_sample0.html",
    "poisson_four_way_rel_sol.html",
    "poisson_cdr_rule_comparison.html",
    "poisson_weak_result_fields_sample0.html",
    "poisson_weak_result_errors_sample0.html",
    "cdr_weak_result_fields_sample9.html",
    "cdr_weak_result_errors_sample9.html",
]


def _slides(source: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?m)^## ", source)[1:]
    return [(part.splitlines()[0].strip(), part) for part in parts]


def _notes(slide: str) -> str:
    match = re.search(r"::: \{\.notes\}\n(.*?)\n:::", slide, flags=re.DOTALL)
    if match is None:
        raise AssertionError("Every logical slide must contain one notes block")
    return match.group(1)


def _visible_body(slide: str) -> str:
    return re.sub(
        r"::: \{\.notes\}\n.*?\n:::",
        "",
        slide,
        flags=re.DOTALL,
    )


def _fragment_count(slide: str) -> int:
    body = _visible_body(slide)
    raw_html = len(re.findall(r'class="fragment(?:\s|\")', body))
    fenced_div = len(re.findall(r"(?m)^:::+ \{[^\n}]*\.fragment(?:\s|\})", body))
    return raw_html + fenced_div


def test_qmd_locks_slide_order_animation_and_language_contract() -> None:
    source = QMD_PATH.read_text(encoding="utf-8")
    slides = _slides(source)

    assert [title for title, _ in slides] == EXPECTED_TITLES
    assert len(slides) == 18

    for index, ((_, slide), expected_fragments) in enumerate(
        zip(slides, EXPECTED_FRAGMENT_COUNTS, strict=True), start=1
    ):
        notes = _notes(slide)
        assert slide.count("::: {.notes}") == 1
        assert _fragment_count(slide) == expected_fragments
        assert notes.count("**Click:**") == expected_fragments
        assert re.search(r"[가-힣]", notes), f"Slide {index} notes must be Korean"
        assert not re.search(r"[가-힣]", _visible_body(slide))
        assert "**Target:" in notes or "**Timing:" in notes
        assert "**If timing is tight:**" in notes
        assert "**Transition:**" in notes

    assert sum(EXPECTED_FRAGMENT_COUNTS) == 44


def test_revision_content_contract_is_versionless_and_layout_ready() -> None:
    source = QMD_PATH.read_text(encoding="utf-8")
    styles = STYLES_PATH.read_text(encoding="utf-8")
    slides = dict(_slides(source))

    assert not re.search(r"\b(?:v4|v5|v6)\b", source, flags=re.IGNORECASE)
    assert not re.search(r"output[- ]contract", source, flags=re.IGNORECASE)

    line_length = slides["Where Line Length Enters Projection and Pull-Back"]
    assert "line-length-geometry" in line_length
    assert "line-length ratio" in line_length
    assert "2.19x" in line_length
    assert "4.80x" in line_length

    projection = slides["Projection Strategies Tested for the Annulus Transition"]
    for definition in ("d_0=", "\\kappa=", "d_{\\mathrm{RPS}}="):
        assert definition in projection
    assert "equal-response baseline" in projection
    assert "attenuation from length imbalance" in projection

    comparison = slides["Three Ways to Choose the Cross-Axis Reconstruction Weight"]
    assert comparison.count("<dt>Signal</dt>") == 3
    assert comparison.count("<dt>Sample adaptive</dt>") == 3
    assert comparison.count("<dt>Operator aware</dt>") == 3
    assert "method-common-invariant" in comparison

    poisson = slides[
        "Pure Poisson: Directional Sources and Weak-Residual Reconstruction"
    ]
    assert "result-formula-card" in poisson
    assert "\\begin{aligned}" in poisson

    response_problem = slides[
        "Exact Balance Does Not Imply Directional Response Consistency"
    ]
    assert "p=\\frac{P}{L_x^2}" in response_problem
    assert "\\mathcal C_f" in response_problem
    assert "\\widetilde\\phi=\\frac{f+d}{2}" in response_problem
    assert "normal / common" in response_problem
    assert "tangent / difference" in response_problem
    assert "m_0=H_x\\widetilde\\phi-H_y\\widetilde\\psi" in response_problem
    assert "Feasible does not mean response-consistent" in response_problem
    assert response_problem.count('class="response-factor operator"') == 3
    assert response_problem.count('class="response-factor context"') == 2

    response_projection = slides[
        "Next Candidate: Tangent Correction Within the Balance Plane"
    ]
    assert "\\phi(\\delta)=\\widetilde\\phi+\\delta" in response_projection
    assert "J(\\delta)=\\frac12" in response_projection
    assert "g=\\nabla J(0)" in response_projection
    assert "(H_x+H_y)^\\top M_\\Omega m_0" in response_projection
    assert "\\gamma_{s,j}^2" in response_projection
    assert "H_s^\\top M_\\Omega H_s" in response_projection
    assert "D_j=G_j+(\\lambda_{\\mathrm{rel}}" in response_projection
    assert "\\delta_j=-\\eta\\frac{g_j}{D_j}" in response_projection
    assert "The column diagonal scales the tangent step" in response_projection
    assert "it does not allocate the raw residual" in response_projection
    assert "g^\\top(-D^{-1}g)\\leq 0" in response_projection
    assert "\\phi+\\psi=f" in response_projection
    assert "A_s=H_s^\\top M_\\Omega H_s" not in response_projection
    assert "full coupled solve" not in response_projection
    assert "\\min_{\\delta\\phi_j+\\delta\\psi_j=r_j}" not in response_projection
    assert "\\delta\\phi_j=\\frac{\\bar\\gamma_{y,j}^2}" not in response_projection
    assert "#next-candidate-tangent-correction-within-the-balance-plane h2" in styles
    assert "font-size: 30pt;" in styles


def test_main_deck_timing_is_advisory_not_a_fixed_total() -> None:
    source = QMD_PATH.read_text(encoding="utf-8")
    main_slides = _slides(source)[:16]

    assert len(main_slides) == 16
    assert "25-minute" not in source
    assert "25분" not in source


def test_slide_plan_matches_the_tangent_method_contract() -> None:
    plan = SLIDE_PLAN_PATH.read_text(encoding="utf-8")

    assert "### Slide 15: Exact Balance Does Not Imply" in plan
    assert "### Slide 16: Tangent Correction Within the Balance Plane" in plan
    assert "g=\\nabla J(0)=(H_x+H_y)^\\top M_\\Omega m_0" in plan
    assert "\\delta_j=-\\eta\\frac{g_j}{D_j}" in plan
    assert "The column diagonal scales the tangent gradient" in plan
    assert "Why Equal Source Correction Is Not Equal Response" not in plan
    assert "Column-Diagonal Green-Response Projection" not in plan


def test_deck_uses_only_local_presentation_assets() -> None:
    source = QMD_PATH.read_text(encoding="utf-8")
    for asset_name in EXPECTED_ASSETS:
        asset_path = DECK_ROOT / "assets" / asset_name
        assert asset_path.is_file()
        assert f'src="assets/{asset_name}"' in source

    assert not (DECK_ROOT / "annulus_transition_error.pdf").exists()


def test_rendered_reveal_html_is_offline_and_structurally_complete() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert len(re.findall(r'<section id="[^"]+" class="slide level2">', html)) == 18
    assert html.count('<aside class="notes">') == 18
    assert len(re.findall(r'class="fragment(?:\s|\")', html)) == 44
    assert not re.search(
        r'<(?:script|link|iframe|img)[^>]+(?:src|href)="https?://',
        html,
    )
    assert "annulus_transition_error_files/libs/revealjs/dist/reveal.js" in html
    assert "assets/plotly.min.js" not in html
    assert not re.search(r"\b(?:v4|v5|v6)\b", html, flags=re.IGNORECASE)
    assert not re.search(r"output[- ]contract", html, flags=re.IGNORECASE)
    assert "Exact Balance Does Not Imply Directional Response Consistency" in html
    assert "Next Candidate: Tangent Correction Within the Balance Plane" in html
    assert "Why Equal Source Correction Is Not Equal Response" not in html
    assert "Next Candidate: Allocate Balance Correction by Green Response" not in html
    assert '<div class="response-factor-list"' in html
    assert (
        '<div class="response-factor-list" aria-label="Directional response factors">\n<p>'
        not in html
    )


def test_browser_qa_covers_final_and_intermediate_states_at_both_viewports() -> None:
    report = json.loads(QA_REPORT_PATH.read_text(encoding="utf-8"))

    assert [item["viewport"] for item in report["viewports"]] == [
        "1600x900",
        "1280x720",
    ]
    for viewport in report["viewports"]:
        assert viewport["slideCount"] == 18
        assert viewport["overflow"] == []
        assert viewport["overlap"] == []
        assert viewport["fragmentStatesChecked"] == 60
        assert viewport["fragmentOverflow"] == []
        assert viewport["fragmentOverlap"] == []
        assert viewport["pageErrors"] == []
        assert viewport["externalRequests"] == []
