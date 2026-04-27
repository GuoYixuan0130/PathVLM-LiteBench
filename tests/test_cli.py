from pathvlm_litebench.cli import main


def test_cli_version(capsys):
    exit_code = main(["version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "PathVLM-LiteBench" in captured.out
    assert "0.1.0" in captured.out


def test_cli_models(capsys):
    exit_code = main(["models"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "clip" in captured.out
    assert "plip" in captured.out
    assert "conch" in captured.out


def test_cli_demos(capsys):
    exit_code = main(["demos"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "examples/01_patch_text_retrieval_demo.py" in captured.out


def test_cli_no_subcommand_shows_help(capsys):
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out
    assert "pathvlm-litebench" in captured.out
