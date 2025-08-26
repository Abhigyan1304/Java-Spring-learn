SQL_PROMPT = """
You are a Text-to-SQL model. Use the context and schema below.

Context:
{context}

Schema:
{schema}

Question:
{question}

Return only the SQL query.
"""
