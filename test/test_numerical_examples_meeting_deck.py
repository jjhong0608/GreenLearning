"""The 45-slide narrative and offline rendering must match the agreed contract."""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs/meeting/numerical_examples"


class Tags(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_exact_slide_order_notes_fragments_and_assets():
    contract = json.loads((DECK / "slide_contract.json").read_text())["slides"]
    qmd = (DECK / "numerical_examples.qmd").read_text()
    sections = re.split(r"(?m)^## ", qmd)[1:]
    assert len(sections) == len(contract) == 45
    assert [
        sum(s["id"].startswith(f"ex{i}-") for s in contract) for i in range(1, 5)
    ] == [8, 10, 10, 12]
    for section, entry in zip(sections, contract, strict=True):
        assert f"#{entry['id']}" in section.splitlines()[0]
        visible, notes = section.split("::: {.notes}")
        assert re.search("[가-힣]", notes)
        assert not re.search("[가-힣ぁ-ゟァ-ヿ]", visible)
        assert not re.search("[ぁ-ゟァ-ヿ]", notes)
        assert entry["korean_notes"] == notes.split(":::")[0].strip()
        if entry["fragments"]:
            assert "클릭" in notes
        for asset in entry["assets"]:
            assert (DECK / asset).is_file()


def test_sample_and_scientific_claim_contract():
    qmd = (DECK / "numerical_examples.qmd").read_text()
    for item in (
        "A Typical Baseline Source {#ex1-sample72}",
        "A Difficult Baseline Source {#ex1-sample88}",
        "A Typical Identity-Baseline Source {#ex2-sample2}",
        "A Difficult Identity-Baseline Source {#ex2-sample27}",
        "Typical Equal-Mean Baseline {#ex3-sample1}",
        "Difficult Equal-Mean Baseline {#ex3-sample34}",
        "sample15",
        "103 / 400",
        "955,994",
        "958,018",
        "1,383,800",
        "1,482,872",
        "K29",
        "not a theorem",
        "not an accuracy guarantee",
    ):
        assert item in qmd
    assert not re.search(r"\bv[56]\b|output.contract.version|normalization", qmd, re.I)


def test_rendered_math_notes_and_local_assets():
    html = (DECK / "numerical_examples.html").read_text()
    tags = Tags(html).tags
    slides = [
        a for t, a in tags if t == "section" and "slide" in a.get("class", "").split()
    ]
    assert len(slides) == 45
    assert (
        sum(t == "aside" and "notes" in a.get("class", "").split() for t, a in tags)
        == 45
    )
    assert sum(t == "math" for t, a in tags) >= 20
    for content in re.findall(r'<span class="math[^\"]*">(.*?)</span>', html, re.S):
        assert "<math" in content, "Unconverted TeX must not pass rendering QA"
    for _, attrs in tags:
        src = attrs.get("src", attrs.get("data-src", ""))
        if not src:
            continue
        assert not src.startswith(("http:", "https:"))
        if not src.startswith("data:"):
            assert (DECK / src).is_file()
    assert "<p>:::</p>" not in html
    assert "<p>$$</p>" not in html


def test_plan_and_notes_are_delivered():
    plan = (ROOT / "docs/meeting/numerical_examples_slide_plan.md").read_text()
    assert "45장" in plan and "1600x900" in plan and "1280x720" in plan
    notes = (DECK / "speaker_notes_ko.md").read_text()
    assert len(re.findall(r"(?m)^## ", notes)) == 45


def test_preconditioned_subspace_explanation():
    qmd = (DECK / "numerical_examples.qmd").read_text()
    section = qmd.split("{#ex2-space}")[1].split("\n## ")[0]
    assert r"D^{-1}AD^{-1}g_0" in section
    assert r"(D^{-1}A)^jD^{-1}g_0" in section
    assert "nondegenerate" in section and "Matrix-free" in section
    assert 'data-fragment-index="0"' in section
    assert 'data-fragment-index="1"' in section


def test_local_hat_definition_precedes_blend():
    qmd = (DECK / "numerical_examples.qmd").read_text()
    section = qmd.split("{#ex3-method}")[1].split("\n## ")[0]
    assert r"\chi_i(x_i)=1" in section
    assert r"\frac{x_{i+1}-x_{i-1}}{2}" in section
    assert "connected segment" in section
    assert section.index('class="hat-definition"') < section.index(
        'data-fragment-index="0"'
    )
    assert section.index('data-fragment-index="0"') < section.index(
        r"u_{\mathrm{equal}}"
    )


def test_transverse_input_is_explicit_and_specialized_for_square():
    contract = json.loads((DECK / "slide_contract.json").read_text())["slides"]
    slide = next(s for s in contract if s["id"] == "ex1-motivation")
    assert slide["fragments"] == [0, 1]
    qmd = (DECK / "numerical_examples.qmd").read_text()
    section = qmd.split("{#ex1-motivation}")[1].split("\n## ")[0]
    for formula in (
        r"\log\frac{L_\perp}{L_{\mathrm{ref}}}",
        r"\log\frac{L_\parallel}{L_\perp}",
        r"\kappa=\frac{4L_\parallel^2 L_\perp^2}{(L_\parallel^2+L_\perp^2)^2}",
        r"\mathbf z_\perp=[t_\perp,0,0,1]",
    ):
        assert formula in section
