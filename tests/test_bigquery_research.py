"""No-credential BigQuery SQL generation tests."""
import pytest

from aptos.reporting.bigquery_research import render_queries


@pytest.mark.pure
def test_sql_is_read_only_and_matches_export_contract():
    q = render_queries("research-project1", "APTOS_2019", "aptos_")
    assert len(q) == 4
    for name, sql in q.items():
        assert sql.startswith("SELECT"), name
        assert "`research-project1.APTOS_2019.aptos_labels`" in sql
        for dangerous in ("CREATE ", "DROP ", "DELETE ", "UPDATE ", "INSERT "):
            assert dangerous not in sql
    assert "diagnosis >= 2" in q["label_integrity"]
    assert "COUNT(DISTINCT split)" in q["repeated_id_across_splits"]


@pytest.mark.pure
@pytest.mark.parametrize("project,dataset,prefix", [
    ("invalid.project", "APTOS_2019", "aptos_"),
    ("abcde1", "unsafe-dataset", "aptos_"),
    ("abcde1", "APTOS_2019", "aptos_`;DROP"),
])
def test_rejects_invalid_sql_identifiers(project, dataset, prefix):
    with pytest.raises(ValueError):
        render_queries(project, dataset, prefix)
