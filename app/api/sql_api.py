from fastapi import APIRouter, HTTPException, Query
from app.services.sql_service import generate_sql
from app.guards.guardrails import validate_question
from app.logger import logger
from app.models.sql_models import SQLResponse

router = APIRouter()

@router.get("/get_sql_query", response_model=SQLResponse)
async def get_sql_query(
    input: str = Query(..., description="User natural language question"),
    country: str = Query(..., description="Country code (e.g. IN, PH, TH)"),
    additional: str = Query(..., description="Additional context (mandatory)")
):
    """
    Generate a SQL query from input + RAG + additional context.
    """
    if not validate_question(input):
        raise HTTPException(status_code=400, detail="Invalid input")

    try:
        sql = await generate_sql(
            question=input,
            schema=additional,
            country=country
        )
        logger.info(f"/sql/get_sql_query -> {sql}")
        return SQLResponse(sql=sql)

    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate SQL")
