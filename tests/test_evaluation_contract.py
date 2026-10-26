import json
from pathlib import Path

def test_golden_dataset_schema(tmp_path):
    data = tmp_path / "golden.jsonl"
    data.write_text(json.dumps({"query":"How do I publish a space?", "relevant_urls":["https://superspaceapp.com/support/article/publish-or-unpublish-a-space"]}), encoding="utf-8")
    row = json.loads(data.read_text(encoding="utf-8"))
    assert row["query"] and row["relevant_urls"]

def test_index_manifest_has_expected_files_when_provided():
    path = Path("data/index")
    if path.exists():
        assert (path / "manifest.json").is_file()
        assert (path / "chunks.json").is_file()
        assert (path / "vectors.npy").is_file()
