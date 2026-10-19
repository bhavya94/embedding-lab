import json

from embedding_lab.cli import main


def test_results_dir_saves_what_the_command_prints(tmp_path, capsys):
    main(["--results-dir", str(tmp_path), "pooling"])
    printed = capsys.readouterr().out
    saved = json.loads((tmp_path / "pooling.json").read_text())
    assert [row["setting"] for row in saved["rows"]] == ["mean pooling", "cls pooling"]
    for row in saved["rows"]:
        assert f"{row['ndcg@10']:.4f}" in printed


def test_without_results_dir_nothing_is_written(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["data"])
    assert "3633 documents" in capsys.readouterr().out
    assert not any(tmp_path.glob("*.json"))
