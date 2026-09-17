"""Job-status endpoint backed by arq's ephemeral Redis metadata.

Job status is deliberately not persisted (plan Key Decision 6): arq keeps each
job's result for ``jobs.keep_result_s`` and then forgets it, so an expired or
unknown id is indistinguishable from a typo and returns ``404``.
"""

from __future__ import annotations

from arq.jobs import Job, JobStatus
from fastapi import APIRouter, HTTPException, status

from app.api.deps import ArqPoolDep
from app.api.schemas.documents import JobStatusOut

router = APIRouter(prefix="/jobs", tags=["jobs"])

#: arq statuses mapped onto the API's public vocabulary.
_STATUS_MAP: dict[JobStatus, str] = {
    JobStatus.deferred: "queued",
    JobStatus.queued: "queued",
    JobStatus.in_progress: "running",
    JobStatus.complete: "done",
}


@router.get("/{job_id}", response_model=JobStatusOut)
async def get_job_status(job_id: str, pool: ArqPoolDep) -> JobStatusOut:
    """Return the status of a job, or ``404`` when it is unknown or expired."""
    job = Job(job_id, pool)
    job_status = await job.status()
    if job_status is JobStatus.not_found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")

    if job_status is not JobStatus.complete:
        return JobStatusOut(status=_STATUS_MAP.get(job_status, "failed"))

    result = await job.result_info()
    if result is not None and not result.success:
        return JobStatusOut(status="failed", error=_error_message(result.result))
    return JobStatusOut(status="done")


def _error_message(result: object) -> str:
    """Render an arq failure payload as a readable message."""
    if isinstance(result, BaseException):
        return f"{type(result).__name__}: {result}"
    return str(result)
