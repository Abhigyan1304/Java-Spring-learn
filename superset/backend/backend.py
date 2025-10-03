from fastapi import FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import uvicorn
import time
from pydantic import BaseModel
import asyncio
import json
import random

# Create a FastAPI app instance
app = FastAPI()

# Configure CORS middleware
# This allows requests from your frontend's development server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# You can use a router to organize your API endpoints
router = APIRouter(prefix="/api")
router_v1 = APIRouter(prefix="/api/v1")


# Hardcoded data for demonstration purposes
HARDCODED_SQL_QUERY = "SELECT date_trunc('month', sale_date) AS sale_month, SUM(amount) AS total_sales FROM sales GROUP BY sale_month ORDER BY sale_month;"
HARDCODED_CHART_DATA = {
    "chart_id": f"chart-{int(time.time())}",
    "dataset_id": f"datasxet-{int(time.time())}",
    "chart_type": "bar_chart",
    "iframeUrl": "https://superset.datatest.ch/superset/explore/p/b1Jx9y6GxR7/?standalone=1&height=400",
    "user_id": "user-12345",
    "user_input": ""
}
HARDCODED_SUGGESTIONS = [
    "What were the total sales by product category?",
    "Compare this month's sales to the same month last year.",
    "Show a breakdown of sales by region.",
    "What is the average order value per customer?"
]
HARDCODED_SUMMARY = "The chart displays a clear trend of increasing total sales over time, with a notable peak in sales during the summer months. This could be due to seasonal product demand or a specific marketing campaign. Further analysis could explore sales by region or product category to identify key growth drivers."


# Use a Pydantic model for type validation and automatic documentation
class UserPrompt(BaseModel):
    user_prompt: str

class SqlQuery(BaseModel):
    sql_query: str

@router.post("/generate-sql")
async def generate_sql(user_prompt_data: UserPrompt):
    """
    Simulates an API endpoint to generate an SQL query from a user prompt.
    """
    # Simulate some processing time asynchronously
    await asyncio.sleep(2)
    # The user_prompt_data object is automatically parsed from the request body
    user_prompt = user_prompt_data.user_prompt
    if not user_prompt:
        return {"error": "User prompt is required"}

    return {"sql_query": HARDCODED_SQL_QUERY}

@router.post("/generate-chart-data")
async def generate_chart_data(sql_query_data: SqlQuery):
    """
    Simulates an API endpoint to generate chart data from an SQL query.
    """
    await asyncio.sleep(2)
    sql_query = sql_query_data.sql_query
    if not sql_query:
        return {"error": "SQL query is required"}

    chart_data = HARDCODED_CHART_DATA.copy()
    chart_data['user_input'] = sql_query

    return chart_data

@router_v1.get("/sse/insights")
async def insights_stream(request: Request):
    """
    Streams a hardcoded summary and suggestions as Server-Sent Events.
    """
    async def event_generator():
        # Yield suggestions first
        for i, suggestion in enumerate(HARDCODED_SUGGESTIONS):
            yield f'data: {json.dumps({"suggestions": [suggestion]})}\n\n'
            # To simulate streaming, wait a short random time between suggestions
            await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Now stream the summary in chunks to simulate a longer generation
        chunk_size = 30
        summary_chunks = [HARDCODED_SUMMARY[i:i + chunk_size] for i in range(0, len(HARDCODED_SUMMARY), chunk_size)]
        
        for chunk in summary_chunks:
            yield f'data: {json.dumps({"summary": chunk})}\n\n'
            await asyncio.sleep(random.uniform(0.1, 0.5))

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Include both routers in the main app
app.include_router(router)
app.include_router(router_v1)

if __name__ == '__main__':
    # To run this with Uvicorn, you can use the command below.
    # Note that the app name is "main:app" because the file is typically
    # named `main.py` and the FastAPI instance is named `app`.
    # You can change this to match your filename and variable name.
    # uvicorn main:app --reload --port 5000
    uvicorn.run(app, host="127.0.0.1", port=5000)
