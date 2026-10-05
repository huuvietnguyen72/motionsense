import csv
import io
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from motionsense_app.dependencies import (
    RegistryDependency,
    StoreDependency,
    require_compatible_dataset,
)
from motionsense_app.models import FEATURE_IDS, Prediction
from motionsense_app.models.prediction import predict_rows
from motionsense_app.recognition import CsvPredictions, SampleRecognitionRequest
from motionsense_app.recognition.csv_input import MAX_UPLOAD_BYTES, validate_csv

router = APIRouter(prefix="/api/recognition")


@router.post("/sample", response_model=Prediction)
def recognize_sample(
    body: SampleRecognitionRequest, store: StoreDependency, registry: RegistryDependency,
) -> dict:
    artifact = registry.load(body.model_id)
    require_compatible_dataset(store, artifact)
    sample = store.sample(body.sample_id)
    result = predict_rows(artifact, [sample["features"]])[0]
    return dict(result, sample_id=sample["sample_id"], actual_label=sample["actual_label"])


@router.post("/csv", response_model=CsvPredictions)
def recognize_csv(
    file: Annotated[UploadFile, File()],
    model_id: Annotated[str, Form(min_length=1)],
    registry: RegistryDependency,
) -> dict:
    artifact = registry.load(model_id)
    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        file.file.close()
    rows, labels = validate_csv(content, artifact["manifest"]["feature_ids"], FEATURE_IDS)
    results = predict_rows(artifact, rows)
    for result, label in zip(results, labels):
        result["actual_label"] = label
    return {"items": results, "row_count": len(results)}


@router.get("/template")
def template(model_id: str, store: StoreDependency, registry: RegistryDependency) -> Response:
    artifact = registry.load(model_id)
    require_compatible_dataset(store, artifact)
    sample = store.sample("test:000001")
    feature_ids = artifact["manifest"]["feature_ids"]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(feature_ids + ["activity"])
    writer.writerow([sample["features"][fid] for fid in feature_ids] + [sample["actual_label"]])
    return Response(
        stream.getvalue().encode("utf-8-sig"), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="motionsense-template.csv"'},
    )
