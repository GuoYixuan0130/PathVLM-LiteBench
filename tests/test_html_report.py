from pathlib import Path

from PIL import Image

from pathvlm_litebench.visualization import save_retrieval_html_report


def test_html_report_uses_relative_image_paths(tmp_path: Path):
    image_dir = tmp_path / "images"
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

    assert "../images/patch_red.png" in html_text
    assert "src='../images/patch_red.png'" in html_text
    assert "src='..\\images\\patch_red.png'" not in html_text
    assert Path(saved_path).exists()
