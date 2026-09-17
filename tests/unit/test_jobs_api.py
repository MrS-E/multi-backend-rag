"""API-level tests for the job-status endpoint.

``GET /jobs/{id}`` reads arq's ephemeral Redis metadata through
``arq.jobs.Job``. The router's call chain (``Job(job_id, pool)`` → ``status()``
→ ``result_info()``) is mirrored by a fake ``Job`` installed at the arq
boundary, so the mapping onto the API's ``queued/running/done/failed``
vocabulary is exercised without Redis.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from arq.jobs import JobResult, JobStatus
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routers


class _FakeJob:
    """Stand-in for ``arq.jobs.Job``: returns a canned status and result."""

    status_value: JobStatus = JobStatus.not_found
    result_value: JobResult | None = None
    instances: list[tuple[str, object]] = []

    def __init__(self, job_id: str, pool: object) -> None:
        self.job_id = job_id
        self.pool = pool
        type(self).instances.append((job_id, pool))

    async def status(self) -> JobStatus:
        return type(self).status_value

    async def result_info(self) -> JobResult | None:
        return type(self).result_value


def _job_result(success: bool, result: object) -> JobResult:
    now = datetime.now(UTC)
    return JobResult(
        function="ingest_events",
        args=(),
        kwargs={},
        job_try=1,
        enqueue_time=now,
        score=None,
        job_id="job-1",
        success=success,
        result=result,
        start_time=now,
        finish_time=now,
        queue_name="arq:queue",
    )


@pytest.fixture
def fake_job(monkeypatch: pytest.MonkeyPatch) -> type[_FakeJob]:
    """Install the fake ``Job`` and reset its recorded state per test."""
    _FakeJob.instances = []
    _FakeJob.status_value = JobStatus.not_found
    _FakeJob.result_value = None
    monkeypatch.setattr(routers.jobs, "Job", _FakeJob)
    return _FakeJob


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """The shared test client, with a non-null arq pool so the dep resolves."""
    app.state.arq_pool = object()
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("arq_status", "expected"),
    [
        (JobStatus.deferred, "queued"),
        (JobStatus.queued, "queued"),
        (JobStatus.in_progress, "running"),
    ],
)
def test_given_pending_arq_state_when_getting_then_mapped_status(
    client: TestClient, fake_job: type[_FakeJob], arq_status: JobStatus, expected: str
) -> None:
    fake_job.status_value = arq_status

    response = client.get("/jobs/job-1")

    assert response.status_code == 200
    assert response.json() == {"status": expected, "error": None}


def test_given_unknown_job_when_getting_then_404(
    client: TestClient, fake_job: type[_FakeJob]
) -> None:
    fake_job.status_value = JobStatus.not_found

    response = client.get("/jobs/expired-id")

    assert response.status_code == 404
    assert response.json()["detail"] == "job not found"


def test_given_complete_job_when_getting_then_done(
    client: TestClient, fake_job: type[_FakeJob]
) -> None:
    fake_job.status_value = JobStatus.complete
    fake_job.result_value = _job_result(success=True, result=3)

    response = client.get("/jobs/job-1")

    assert response.status_code == 200
    assert response.json() == {"status": "done", "error": None}


def test_given_failed_job_when_getting_then_failed_with_error(
    client: TestClient, fake_job: type[_FakeJob]
) -> None:
    fake_job.status_value = JobStatus.complete
    fake_job.result_value = _job_result(success=False, result=RuntimeError("boom"))

    response = client.get("/jobs/job-1")

    assert response.status_code == 200
    assert response.json() == {"status": "failed", "error": "RuntimeError: boom"}


def test_given_failed_job_with_string_result_when_getting_then_failed_with_message(
    client: TestClient, fake_job: type[_FakeJob]
) -> None:
    fake_job.status_value = JobStatus.complete
    fake_job.result_value = _job_result(success=False, result="provider down")

    response = client.get("/jobs/job-1")

    assert response.status_code == 200
    assert response.json() == {"status": "failed", "error": "provider down"}


def test_given_job_lookup_when_getting_then_pool_passed_through(
    client: TestClient, fake_job: type[_FakeJob], app: FastAPI
) -> None:
    fake_job.status_value = JobStatus.queued

    client.get("/jobs/job-42")

    assert fake_job.instances == [("job-42", app.state.arq_pool)]


def test_given_no_arq_pool_when_getting_then_503(app: FastAPI) -> None:
    with TestClient(app) as test_client:
        response = test_client.get("/jobs/job-1")

    assert response.status_code == 503
