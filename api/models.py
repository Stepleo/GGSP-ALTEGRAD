from pydantic import BaseModel

class CheckResultsRequest(BaseModel):
    run_id: str  # The run ID of the saved models in MLflow
class CheckResultsResponse(BaseModel):
    metrics: dict
