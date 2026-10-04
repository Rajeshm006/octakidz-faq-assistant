from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .faq_data import load_faq_dataset
from .schemas import ChatRequest, FinalOutput
from .service import FAQAssistantService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Failing here is intentional: the API must not run without its validated FAQ data/key.
    settings: Settings = get_settings()
    faq_dataset = load_faq_dataset()
    app.state.service = FAQAssistantService(faq_dataset, settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="OctaKidz FAQ Assistant API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, __: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": "Request validation failed."})

    @app.exception_handler(Exception)
    async def safe_error_handler(_: Request, __: Exception):
        return JSONResponse(status_code=500, content={"detail": "Unable to process this request safely."})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=FinalOutput)
    def chat(payload: ChatRequest) -> FinalOutput:
        # This endpoint is intentionally synchronous. FastAPI runs it in a worker
        # thread, allowing CrewAI's synchronous kickoff to execute without nesting
        # inside the server event loop and without blocking other async requests.
        try:
            return app.state.service.process_message(payload.message, payload.session_id)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail="Unable to process this message safely.") from exc

    return app


app = create_app()
