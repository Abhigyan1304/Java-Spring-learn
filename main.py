# Standard library imports
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import DictCursor
import sqlparse
from dataclasses import dataclass

# Database configuration
@dataclass
class DBConfig:
    """Database configuration for PostgreSQL connection.
    
    Attributes:
        host: Database server hostname
        port: Database server port
        database: Name of the database
        user: Database user
        password: Database password
    """
    host: str = "localhost"  # Change this to your database host
    port: int = 5432        # Default PostgreSQL port
    database: str = "postgres"  # Change this to your database name
    user: str = "postgres"      # Change this to your database user
    password: str = ""          # Add your database password here

# Initialize database configuration
db_config = DBConfig()

# Unsloth imports for fast LLM training
from unsloth import FastLanguageModel, PatchFastRL

# Patch the FastLanguageModel with GRPO (Generative Reinforcement Policy Optimization)
# This is required for the GRPO training process
PatchFastRL("GRPO", FastLanguageModel)  # needed for GRPO


from unsloth import is_bfloat16_supported  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
import re  # noqa: E402
from datasets import load_dataset, Dataset  # noqa: E402
from vllm import SamplingParams  # noqa: E402

from dataclasses import dataclass  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from dataclasses import field  # noqa: E402
import hydra  # noqa: E402

max_seq_length = 1024  # Can increase for longer reasoning traces
lora_rank = 64  # Larger rank = smarter, but slower


# Configuration class for LoRA (Low-Rank Adaptation) parameters
@dataclass
class LoraConfig:
    # Rank of the LoRA matrices (higher rank = more capacity but slower training)
    rank: int = 64
    
    # List of model layers to apply LoRA to
    # These are the attention and feed-forward layers in the transformer
    target_modules: List = field(
        default_factory=lambda: [
            "q_proj",  # Query projection
            "k_proj",  # Key projection
            "v_proj",  # Value projection
            "o_proj",  # Output projection
            "gate_proj",  # Gate projection for MLP
            "up_proj",   # Upward projection for MLP
            "down_proj", # Downward projection for MLP
        ]
    )
    # Gradient checkpointing strategy for memory efficiency
    use_gradient_checkpointing: str = "unsloth"
    # Random seed for reproducibility
    random_state: int = 3407


# Configuration class for the main model settings
@dataclass
class ModelConfig:
    # Maximum sequence length for input text (affects memory usage)
    max_seq_length: int = 1024
    
    # Enable 4-bit quantization for reduced memory usage
    load_in_4bit: bool = True
    
    # Use vLLM's optimized inference
    fast_inference: bool = True
    
    # LoRA configuration for efficient fine-tuning
    lora: LoraConfig = field(default_factory=lambda: LoraConfig())

    # Control GPU memory usage (0.0 to 1.0)
    gpu_memory_utilization: float = 0.5


def prepare_model(cfg: DictConfig):
    """Initialize and configure the Qwen model with LoRA for fine-tuning.
    
    This function:
    1. Loads the base Qwen3-0.6B model
    2. Applies quantization and optimization settings
    3. Sets up LoRA for efficient fine-tuning
    
    Args:
        cfg (DictConfig): Configuration object containing model settings
    
    Returns:
        tuple: (model, tokenizer) - The prepared model and its tokenizer
    """
    # Load the base model with optimization settings
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-0.6B",  # Base model to fine-tune
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,  # Use 4-bit quantization
        fast_inference=cfg.model.fast_inference,  # Enable vLLM optimizations
        max_lora_rank=cfg.model.lora.rank,  # Maximum LoRA rank
        gpu_memory_utilization=0.5,  # GPU memory usage (adjust if OOM)
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.model.lora.rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=cfg.model.lora.rank,
        use_gradient_checkpointing=cfg.model.lora.use_gradient_checkpointing,  # Enable long context finetuning
        random_state=cfg.model.lora.random_state,
    )
    return model, tokenizer


## dataset is now loaded above using get_text2sql_dataset

# Text-to-SQL prompt and answer format
# This defines the structure that the model should follow when generating responses
SYSTEM_PROMPT = """
Respond in the following format:
<think>  # First, show the thought process for understanding the question
...
</think>
<sql>    # Then, provide the SQL query that answers the question
...
</sql>
<explanation>  # Finally, explain why this SQL query answers the question
...
</explanation>
"""

SQL_FORMAT = """\
<think>
{think}
</think>
<sql>
{sql}
</sql>
<explanation>
{explanation}
</explanation>
"""

def extract_sql_answer(text: str) -> dict:
    """Extracts <think>, <sql>, <explanation> sections from the model's response.
    
    Args:
        text (str): The full response text from the model
    
    Returns:
        dict: A dictionary containing three keys:
            - 'think': The reasoning process
            - 'sql': The SQL query
            - 'explanation': The explanation of the query
    """
    def extract_tag(tag):
        # Helper function to extract content between XML-style tags
        start = text.find(f"<{tag}>")
        end = text.find(f"</{tag}>")
        if start == -1 or end == -1:  # If tags not found
            return ""
        return text[start + len(f"<{tag}>"):end].strip()
    return {
        "think": extract_tag("think"),
        "sql": extract_tag("sql"),
        "explanation": extract_tag("explanation"),
    }

# Load custom JSON dataset for text-to-SQL
def get_text2sql_dataset(json_path: str) -> Dataset:
    """Loads and preprocesses the text-to-SQL dataset from a JSON file.
    
    The JSON file should contain an array of objects with 'question' and 'answer' fields.
    Each answer should follow the format:
    <think>reasoning</think>
    <sql>query</sql>
    <explanation>details</explanation>
    
    Args:
        json_path (str): Path to the JSON dataset file
    
    Returns:
        Dataset: A HuggingFace dataset with formatted examples
    """
    # Load the JSON file as a HuggingFace dataset
    data = load_dataset("json", data_files=json_path)["train"]
    
    def format_example(x):
        # Format each example as a chat conversation with system prompt
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": x["answer"],
        }
    
    # Apply the formatting to all examples
    data = data.map(format_example)
    return data

# Example usage: replace with your actual dataset path
DATASET_PATH = "path/to/your/text2sql_dataset.json"  # TODO: update this path
dataset = get_text2sql_dataset(DATASET_PATH)

# Placeholder for SQL testing and evaluation
def normalize_sql(sql: str) -> str:
    """Normalize SQL query for comparison by removing extra whitespace and standardizing syntax.
    
    Args:
        sql (str): SQL query to normalize
    
    Returns:
        str: Normalized SQL query
    """
    # Parse and format the SQL query
    parsed = sqlparse.parse(sql)[0]
    normalized = sqlparse.format(str(parsed),
                               strip_comments=True,
                               reindent=True,
                               keyword_case='upper')
    return normalized

def execute_sql_query(sql: str) -> tuple[bool, list[Dict[str, Any]], str]:
    """Execute a SQL query and return results.
    
    Args:
        sql (str): SQL query to execute
    
    Returns:
        tuple: (success, results, error_message)
            - success: True if query executed successfully
            - results: List of dictionaries containing query results
            - error_message: Error description if query failed
    """
    try:
        with psycopg2.connect(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password
        ) as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(sql)
                results = [dict(row) for row in cur.fetchall()]
                return True, results, ""
    except psycopg2.Error as e:
        return False, [], str(e)

def compare_query_results(actual_results: list, expected_results: list) -> float:
    """Compare actual query results with expected results.
    
    Scoring criteria:
    - Exact match: 1.0
    - Same number of rows but slight differences: 0.5-0.9
    - Different number of rows but some matches: 0.1-0.4
    - No matches: 0.0
    
    Args:
        actual_results: List of dictionaries from actual query
        expected_results: List of dictionaries from expected query
    
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    if not actual_results or not expected_results:
        return 0.0
    
    # Convert results to sets of frozensets for comparison
    actual_sets = {frozenset(d.items()) for d in actual_results}
    expected_sets = {frozenset(d.items()) for d in expected_results}
    
    # Calculate exact matches
    exact_matches = len(actual_sets & expected_sets)
    
    # Calculate score based on matches and total size
    total_expected = len(expected_sets)
    total_actual = len(actual_sets)
    
    if exact_matches == total_expected == total_actual:
        return 1.0
    
    # Partial matching score
    precision = exact_matches / total_actual if total_actual > 0 else 0
    recall = exact_matches / total_expected if total_expected > 0 else 0
    
    if precision == 0 or recall == 0:
        return 0.1 * max(precision, recall)
    
    # F1 score scaled to 0.5-0.9 range
    f1 = 2 * (precision * recall) / (precision + recall)
    return 0.5 + (0.4 * f1)

def test_sql_query(sql: str, expected_results: list = None) -> tuple[bool, float, str]:
    """Test if a generated SQL query is valid and produces expected results.
    
    This function:
    1. Validates SQL syntax
    2. Executes the query safely
    3. Compares results with expected output
    4. Provides detailed feedback
    
    Args:
        sql (str): The SQL query to test
        expected_results: List of dictionaries with expected query results
    
    Returns:
        tuple: (is_valid, score, message)
            - is_valid: True if query executed successfully
            - score: Similarity score between 0.0 and 1.0
            - message: Feedback or error message
    """
    try:
        # Normalize and validate SQL syntax
        normalized_sql = normalize_sql(sql)
        
        # Execute the query
        success, results, error = execute_sql_query(normalized_sql)
        
        if not success:
            return False, 0.0, f"SQL Error: {error}"
        
        # Compare results if expected results provided
        if expected_results:
            score = compare_query_results(results, expected_results)
            return True, score, "Query executed successfully"
        
        # If no expected results, just check execution
        return True, 0.5, "Query executed but no expected results for comparison"
        
    except Exception as e:
        return False, 0.0, f"Error: {str(e)}"

def evaluate_reasoning(thinking: str) -> float:
    """Evaluate the quality of the reasoning in the <think> section.
    
    Criteria:
    - Identifies key entities and relationships
    - Shows understanding of the question
    - Explains approach to query construction
    
    Args:
        thinking (str): Content of the <think> section
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    score = 0.0
    # Check for key components in reasoning
    if "table" in thinking.lower() or "column" in thinking.lower():
        score += 0.3  # Identifies database structures
    if "where" in thinking.lower() or "condition" in thinking.lower():
        score += 0.3  # Shows filtering logic
    if "join" in thinking.lower() or "relationship" in thinking.lower():
        score += 0.2  # Understands table relationships
    if len(thinking.split()) >= 20:
        score += 0.2  # Provides detailed explanation
    return min(1.0, score)

def evaluate_sql_answer(answer: dict, expected_results: list = None) -> float:
    """Evaluate the quality of the model's complete answer.
    
    This function evaluates:
    1. SQL query correctness and results (50%)
    2. Quality of reasoning in <think> section (25%)
    3. Accuracy and completeness of explanation (25%)
    
    Args:
        answer (dict): Dictionary containing 'think', 'sql', and 'explanation'
        expected_results: List of dictionaries with expected query results
    
    Returns:
        float: Score between 0.0 and 1.0 indicating answer quality
    """
    # Test SQL query execution and results
    is_valid, sql_score, message = test_sql_query(answer['sql'], expected_results)
    if not is_valid:
        print(f"SQL Evaluation Message: {message}")
        sql_score = 0.0
    
    # Evaluate reasoning quality
    reasoning_score = evaluate_reasoning(answer['think'])
    
    # Evaluate explanation (simple length-based heuristic for now)
    explanation_score = min(1.0, len(answer['explanation'].split()) / 50)
    
    # Calculate weighted final score
    final_score = (
        sql_score * 0.5 +           # SQL correctness: 50%
        reasoning_score * 0.25 +     # Reasoning quality: 25%
        explanation_score * 0.25     # Explanation quality: 25%
    )
    
    return final_score


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that evaluates SQL query correctness.
    
    This function compares the generated SQL queries with the expected answers,
    providing a binary reward (2.0 for exact matches, 0.0 otherwise).
    
    Args:
        prompts: List of input prompts
        completions: List of model-generated completions
        answer: List of expected answers
        **kwargs: Additional arguments
    
    Returns:
        list[float]: List of reward scores (2.0 for correct, 0.0 for incorrect)
    """
    # Extract model responses
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    
    # Extract SQL queries from responses and expected answers
    extracted_responses = [extract_sql_answer(r)["sql"] for r in responses]
    expected_sql = [extract_sql_answer(a)["sql"] if isinstance(a, str) else a.get("sql", "") for a in answer]
    
    # Print debugging information
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nExpected SQL:\n{expected_sql[0]}",
        f"\nModel Response:\n{responses[0]}",
        f"\nExtracted SQL:\n{extracted_responses[0]}",
    )
    
    # Return rewards based on exact SQL matches
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, expected_sql)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if SQL queries follow basic SQL syntax.
    
    Args:
        completions: List of model-generated completions
        **kwargs: Additional arguments
    
    Returns:
        list[float]: 0.5 for each response containing SELECT, 0.0 otherwise
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_sqls = [extract_sql_answer(r)["sql"] for r in responses]
    # Reward responses that contain basic SQL SELECT statements
    return [0.5 if "SELECT" in sql.upper() else 0.0 for sql in extracted_sqls]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that enforces strict formatting rules.
    
    Checks if the response follows exact formatting:
    - Must start with <reasoning> on a new line
    - Must have content between tags
    - Must have proper closing tags with newlines
    - Must end with </answer> on a new line
    
    Args:
        completions: List of model-generated completions
        **kwargs: Additional arguments
    
    Returns:
        list[float]: 0.5 for perfectly formatted responses, 0.0 otherwise
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that allows more flexible formatting.
    
    More lenient than strict_format_reward_func:
    - Allows tags to be on same line as content
    - Allows variable whitespace between sections
    - Doesn't require newlines
    - Just ensures basic tag structure is present
    
    Args:
        completions: List of model-generated completions
        **kwargs: Additional arguments
    
    Returns:
        list[float]: 0.5 for responses with correct tags in order, 0.0 otherwise
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """Calculate a score based on proper XML tag placement and formatting.
    
    Rewards proper tag usage with specific formatting:
    - Each correct tag placement adds 0.125 to the score
    - Penalizes extra content after closing tags
    - Expects newlines around tags for clean formatting
    
    Score breakdown:
    - <reasoning> with newline: +0.125
    - </reasoning> with newlines: +0.125
    - <answer> with newlines: +0.125
    - </answer> with newline: +0.125
    - Penalties for trailing content after tags
    
    Args:
        text (str): The response text to evaluate
    
    Returns:
        float: Score between 0.0 and 0.5 based on formatting quality
    """
    count = 0.0
    # Check for properly formatted opening reasoning tag
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    # Check for properly formatted closing reasoning tag
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    # Check for properly formatted opening answer tag
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        # Penalize for content after the answer section
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    # Check for properly formatted closing answer tag
    if text.count("\n</answer>") == 1:
        count += 0.125
        # Additional penalty for content after final tag
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def strawberry_example(tokenizer, model):
    """Test the base model (without LoRA) using a simple question.
    
    This function demonstrates:
    1. How to format a prompt using the tokenizer
    2. How to set generation parameters
    3. How to generate text using the base model
    
    Args:
        tokenizer: The model's tokenizer
        model: The base language model
    """
    # Format the prompt using the chat template
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "How many r's are in strawberry?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Configure text generation parameters
    sampling_params = SamplingParams(
        temperature=0.8,  # Controls randomness (higher = more random)
        top_p=0.95,      # Nucleus sampling threshold
        max_tokens=1024, # Maximum length of generated text
    )
    
    # Generate response using the base model (no LoRA)
    output = (
        model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=None,  # No LoRA weights used
        )[0]
        .outputs[0]
        .text
    )

    print(output)


# output


def strawberry_example_lora(tokenizer, model):
    """Test the model with LoRA weights using a simple question.
    
    This function demonstrates:
    1. How to use the system prompt
    2. How to generate text using LoRA-adapted weights
    3. Differences from base model generation
    
    Args:
        tokenizer: The model's tokenizer
        model: The fine-tuned language model
    """
    # Format prompt with system prompt and user question
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},  # Include format instructions
            {"role": "user", "content": "How many r's are in strawberry?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Use same sampling parameters as base model for comparison
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    
    # Generate response using the LoRA-adapted model
    output = (
        model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=model.load_lora("grpo_saved_lora"),  # Load and apply LoRA weights
        )[0]
        .outputs[0]
        .text
    )

    print(output)


def save(cfg, model, tokenizer):
    """Save the trained model in multiple formats and optionally push to Hugging Face Hub.
    
    This function handles two types of saving:
    1. GGUF format: Optimized format for deployment
    2. Merged format: Combined base model and LoRA weights
    
    For each format, it can:
    - Save locally to specified directory
    - Push to Hugging Face Hub if token is provided
    
    Args:
        cfg: Configuration object containing saving settings
        model: The trained model to save
        tokenizer: The model's tokenizer
    """
    # Save in GGUF format if enabled
    if cfg.saving.save_gguf.enabled:
        for quant_method in cfg.saving.save_gguf.quantization_methods:
            # Save locally in GGUF format with specified quantization
            model.save_pretrained_gguf(
                cfg.saving.model_dir, tokenizer, quantization_method=quant_method
            )
            # Push to Hugging Face Hub if token is available
            if cfg.saving.token:
                model.push_to_hub_gguf(
                    cfg.saving.hub_model_id,
                    tokenizer,
                    quantization_method=quant_method,
                    token=cfg.saving.token,
                )

    # Save merged model (base + LoRA) if enabled
    if cfg.saving.save_merged.enabled:
        for save_method in cfg.saving.save_merged.methods:
            # Save locally with merged weights
            model.save_pretrained_merged(
                cfg.saving.model_dir, tokenizer, save_method=save_method
            )
            # Push to Hugging Face Hub if token is available
            if cfg.saving.token:
                model.push_to_hub_merged(
                    cfg.saving.hub_model_id,
                    tokenizer,
                    save_method=save_method,
                    token=cfg.saving.token,
                )


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    model, tokenizer = prepare_model(cfg)
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=cfg.training.learning_rate,
        adam_beta1=cfg.training.adam_beta1,
        adam_beta2=cfg.training.adam_beta2,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_generations=cfg.training.num_generations,
        max_prompt_length=cfg.training.max_prompt_length,
        max_completion_length=cfg.training.max_completion_length,
        max_steps=cfg.training.max_steps,
        save_steps=cfg.training.save_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        report_to=cfg.training.report_to,
        output_dir=cfg.training.output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    strawberry_example(tokenizer=tokenizer, model=model)
    strawberry_example_lora(tokenizer=tokenizer, model=model)
    trainer.save_model('/workspace/saved_model')

    if cfg.saving is not None:
        save(cfg, model, tokenizer)


if __name__ == "__main__":
    main()
