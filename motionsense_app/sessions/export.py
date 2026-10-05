import csv
import io

from motionsense_app.sessions.repository import SessionRepository

HEADER = ["session_id", "model_id", "dataset_id", "split", "subject_id", "ordinal",
          "sample_id", "predicted_label", "actual_label",
          *[f"probability_{label}" for label in range(1, 7)], "processed_at"]


def export_session(repo: SessionRepository, id: str) -> bytes:
    session, rows = repo.export_snapshot(id)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(HEADER)
    for row in rows:
        prediction = row["prediction"]
        probabilities = {p["label_id"]: p["value"] for p in prediction["probabilities"]}
        writer.writerow([
            *[session[key] for key in HEADER[:5]], row["ordinal"], row["sample_id"],
            prediction["predicted_label"], prediction["actual_label"],
            *[probabilities[label] for label in range(1, 7)], row["processed_at"],
        ])
    return stream.getvalue().encode("utf-8-sig")
