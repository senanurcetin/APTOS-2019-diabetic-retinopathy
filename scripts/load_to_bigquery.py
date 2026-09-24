"""Load the prepared APTOS-2019 label CSVs into BigQuery.

Optional. Nothing in the pipeline needs BigQuery; this exists for anyone who
wants the labels there.

Prerequisites:
    pip install -e ".[bigquery]"
    python -m aptos.pipeline run prepare
    Authentication: gcloud auth application-default login
                    or  export GOOGLE_APPLICATION_CREDENTIALS=.../key.json

Usage:
    python scripts/load_to_bigquery.py --project PROJECT_ID [--dataset APTOS_2019]

The defaults used to be dataset `aptos2019` with no table prefix, producing
`{project}.aptos2019.labels` - while train.py and aptos.data.labels query
`{project}.APTOS_2019.aptos_labels`. With the defaults, the two could never
meet, and nothing said so. They now agree, and the dataset follows the same
APTOS_BQ_DATASET variable the readers use.
"""
import argparse
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
BQ_DIR = ROOT / "data" / "bq"


def _schema():
    # Imported here, not at module level, so --help works without the
    # google-cloud-bigquery package or any credentials.
    from google.cloud import bigquery

    field = bigquery.SchemaField
    return [
        field("id_code", "STRING", mode="REQUIRED",
              description="Kaggle image id; the file is <id_code>.png"),
        field("diagnosis", "INT64", mode="REQUIRED",
              description="ICDRSS DR severity grade, 0-4"),
        field("diagnosis_label", "STRING", mode="REQUIRED",
              description="Human-readable grade"),
        field("is_referable", "BOOL", mode="REQUIRED",
              description="diagnosis >= 2, referable DR"),
        field("split", "STRING", mode="REQUIRED",
              description="train / valid / test"),
        field("image_file", "STRING", mode="REQUIRED"),
        field("image_uri", "STRING", mode="REQUIRED",
              description="Full gs:// path to the image"),
    ]

# table name -> source CSV
TABLES = {
    "labels": "aptos_labels.csv",
    "train": "train.csv",
    "valid": "valid.csv",
    "test": "test.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=os.environ.get("APTOS_GCP_PROJECT"),
                    help="defaults to $APTOS_GCP_PROJECT")
    ap.add_argument("--dataset", default=os.environ.get("APTOS_BQ_DATASET", "APTOS_2019"))
    ap.add_argument("--location", default="EU", help="EU, US, europe-west1, ...")
    ap.add_argument("--prefix", default="aptos_",
                    help="table name prefix; aptos_ gives the aptos_labels table the readers query")
    args = ap.parse_args()
    if not args.project:
        ap.error("--project is required (or set APTOS_GCP_PROJECT)")

    from google.cloud import bigquery

    client = bigquery.Client(project=args.project)

    ds_ref = bigquery.Dataset(f"{args.project}.{args.dataset}")
    ds_ref.location = args.location
    ds_ref.description = ("APTOS-2019 Blindness Detection labels "
                          "(Kaggle: mariaherrerot/aptos2019)")
    dataset = client.create_dataset(ds_ref, exists_ok=True)
    print(f"dataset ready: {dataset.full_dataset_id} ({dataset.location})")

    for table, csv_name in TABLES.items():
        path = BQ_DIR / csv_name
        if not path.exists():
            raise SystemExit(f"{path} not found - run scripts/prepare_bq_csv.py first")

        table_id = f"{args.project}.{args.dataset}.{args.prefix}{table}"
        job_config = bigquery.LoadJobConfig(
            schema=_schema(),
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            # 3662 rows is small; clustering pays off later when joining
            # against prediction tables.
            clustering_fields=["split", "diagnosis"],
        )
        with path.open("rb") as fh:
            client.load_table_from_file(fh, table_id, job_config=job_config).result()

        loaded = client.get_table(table_id)
        print(f"  {table_id}: {loaded.num_rows} rows")

    q = f"""
        SELECT split, diagnosis, diagnosis_label, COUNT(*) AS n
        FROM `{args.project}.{args.dataset}.{args.prefix}labels`
        GROUP BY split, diagnosis, diagnosis_label
        ORDER BY split, diagnosis
    """
    print("\nverification query:")
    for row in client.query(q).result():
        print(f"  {row.split:<6} {row.diagnosis} {row.diagnosis_label:<17} {row.n}")


if __name__ == "__main__":
    main()
