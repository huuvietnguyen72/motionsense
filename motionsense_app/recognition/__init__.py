"""Recognition of stored samples and validated external CSV files."""

from pydantic import Field

from motionsense_app.models import Contract, Prediction


class SampleRecognitionRequest(Contract):
    model_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)


class CsvPredictions(Contract):
    items: list[Prediction]
    row_count: int = Field(ge=1, le=5000)
