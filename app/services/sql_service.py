from app.services.rag_service import rag_retrieve
from app.helpers.llm_utils import chat_complete
from app.rag.prompt_templates import SQL_PROMPT
from app.guards.guardrails import validate_question, validate_sql, sanitize_sql
from app.logger import logger

async def generate_sql(question: str, schema: str, country: str) -> str:
    """
    Generate SQL using LLM + RAG + Guardrails.
    """
    if not validate_question(question):
        return "/* Invalid question */"

    ctx_chunks = await rag_retrieve(question, country)
    context = "\n".join(ctx_chunks)

    prompt = SQL_PROMPT.format(
        context=context,
        schema=schema or "(schema unknown)",
        question=question
    )

    messages = [
        {"role": "system", "content": "You are a helpful Text-to-SQL assistant."},
        {"role": "user", "content": prompt}
    ]

    out = await chat_complete(messages)
    sql = _strip_sql(out)
    sql = sanitize_sql(sql)

    if not validate_sql(sql):
        return "/* Unsafe SQL blocked */"

    return sql


def _strip_sql(text: str) -> str:
    """
    Naive strip function to extract SQL from LLM output.
    """
    return text.strip().split("```sql")[-1].split("```")[0].strip()
