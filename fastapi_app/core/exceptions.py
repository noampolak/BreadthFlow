from fastapi import HTTPException


class BreadthFlowException(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class PipelineNotFoundError(BreadthFlowException):
    def __init__(self, pipeline_id: str):
        super().__init__(status_code=404, detail=f"Pipeline {pipeline_id} not found")


class PipelineAlreadyRunningError(BreadthFlowException):
    def __init__(self):
        super().__init__(status_code=400, detail="Pipeline is already running")


class SignalNotFoundError(BreadthFlowException):
    def __init__(self, signal_id: str):
        super().__init__(status_code=404, detail=f"Signal {signal_id} not found")


class InfrastructureServiceError(BreadthFlowException):
    def __init__(self, service_name: str):
        super().__init__(status_code=503, detail=f"Service {service_name} is unavailable")
