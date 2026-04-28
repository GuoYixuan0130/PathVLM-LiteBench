from pathlib import Path

from PIL import Image

from pathvlm_litebench.visualization import save_retrieval_html_report


def test_html_report_uses_relative_image_paths(tmp_path: Path):
    image_dir = tmp_path / "external_images"
    output_dir = tmp_path / "outputs"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "patch_red.png"
    Image.new("RGB", (32, 32), color="red").save(image_path)

    output_html_path = output_dir / "report.html"

    saved_path = save_retrieval_html_report(
        prompts=["a red image"],
        retrieval_results=[
            [{"index": 0, "score": 0.9, "path": str(image_path)}]
        ],
        output_html_path=output_html_path,
    )

    html_text = Path(saved_path).read_text(encoding="utf-8")

    assets_dir = output_dir / "report_assets"

    assert Path(saved_path).exists()
    assert assets_dir.exists()
    assert any(path.suffix == ".png" for path in assets_dir.iterdir())
    assert "report_assets/" in html_text
    assert "src='report_assets/" in html_text
    assert "\\\\" not in html_text


def test_html_report_without_copy_images_uses_relative_source_path(tmp_path: Path):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "outputs"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "patch_blue.png"
    Image.new("RGB", (32, 32), color="blue").save(image_path)

    output_html_path = output_dir / "report.html"

    saved_path = save_retrieval_html_report(
        prompts=["a blue image"],
        retrieval_results=[
            [{"index": 0, "score": 0.8, "path": str(image_path)}]
        ],
        output_html_path=output_html_path,
        copy_images=False,
    )

    html_text = Path(saved_path).read_text(encoding="utf-8")
    assert "../images/patch_blue.png" in html_text


def test_html_report_renders_label_target_match_fields(tmp_path: Path):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "outputs"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "patch_hp.png"
    Image.new("RGB", (32, 32), color="green").save(image_path)

    output_html_path = output_dir / "report.html"

    saved_path = save_retrieval_html_report(
        prompts=["a histopathology image of hyperplastic polyp"],
        retrieval_results=[
            [
                {
                    "index": 0,
                    "score": 0.91,
                    "path": str(image_path),
                    "label": "HP",
                    "target_label": "HP",
                    "is_positive": True,
                }
            ]
        ],
        output_html_path=output_html_path,
    )

    html_text = Path(saved_path).read_text(encoding="utf-8")
    assets_dir = output_dir / "report_assets"

    assert Path(saved_path).exists()
    assert assets_dir.exists()
    assert any(path.suffix == ".png" for path in assets_dir.iterdir())
    assert "Label:" in html_text
    assert "HP" in html_text
    assert "Target:" in html_text
    assert "Match:" in html_text
    assert "yes" in html_text
    assert "match-yes" in html_text
