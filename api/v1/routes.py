from fastapi import APIRouter
from api.v1.agent_routes import router as agent_routes


router = APIRouter()
router.include_router(agent_routes, prefix="/agents", tags=["agents"])
