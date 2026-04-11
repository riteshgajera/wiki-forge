"""Agents API router — trigger pipeline runs from the web UI."""
from __future__ import annotations
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from kb.utils.config import settings
from kb.utils.logging import get_logger

router = APIRouter()
logger = get_logger("web.agents")

_running: dict[str, str] = {}


@router.post("/compile")
async def trigger_compile(background_tasks: BackgroundTasks, force: bool = False) -> JSONResponse:
    import uuid
    job_id = str(uuid.uuid4())[:8]
    _running[job_id] = "running"
    background_tasks.add_task(_run_compile, job_id, force)
    return JSONResponse({"job_id": job_id, "status": "started"})


@router.get("/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    status = _running.get(job_id, "not_found")
    return JSONResponse({"job_id": job_id, "status": status})


@router.get("/status")
async def pipeline_status() -> JSONResponse:
    from kb.storage.metadata_store import MetadataStore
    store = MetadataStore(settings.db_path)
    return JSONResponse({"stats": store.stats()})


async def _run_compile(job_id: str, force: bool) -> None:
    try:
        from kb.pipelines.orchestrator import Orchestrator
        orch = Orchestrator(cfg=settings)
        stats = orch.compile(force=force)
        _running[job_id] = f"done:{stats}"
        logger.info("web_compile_done", job_id=job_id, stats=stats)
    except Exception as e:
        _running[job_id] = f"error:{e}"
        logger.error("web_compile_error", job_id=job_id, error=str(e))
