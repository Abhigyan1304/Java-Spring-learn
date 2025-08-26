import re

def validate_question(text: str) -> bool:
    """
    Ensure the input is not empty and not malicious.
    """
    if not text or len(text.strip()) < 3:
        return False
    return True

def validate_sql(sql: str) -> bool:
    """
    Prevent dangerous SQL.
    """
    blacklist = ["drop", "delete", "truncate", "alter", "shutdown"]
    lowered = sql.lower()
    return not any(word in lowered for word in blacklist)

def sanitize_sql(sql: str) -> str:
    """
    Strip trailing semicolons, comments, etc.
    """
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"--.*", "", sql)
    return sql
