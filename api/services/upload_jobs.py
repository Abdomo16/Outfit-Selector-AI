"""In-memory state of photo upload jobs, shared by the worker and the routes."""
import threading

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


class JobNotFound(LookupError):
    pass


class NotYourJob(PermissionError):
    pass


def set_job(job_id: str, **fields) -> None:
    with _LOCK:
        job = _JOBS.get(job_id, {})
        job.update(fields)
        _JOBS[job_id] = job


def get_job(job_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def take_held(job_id: str, user_id: str) -> dict:
    """Hand over the job's held items to [user_id] and clear its pending list."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        if job.get("user_id") != user_id:
            raise NotYourJob(job_id)
        held = job.get("held") or {}
        job["held"] = {}
        job["pending"] = []
        return held
