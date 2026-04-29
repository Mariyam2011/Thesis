"""
prompts.py
----------
Prompt templates for each reasoning strategy.

All prompts use a consistent structure so that token-level entropy
comparisons across strategies are meaningful.
"""


def cot_prompt(question: str) -> str:
    """
    Chain-of-Thought: ask the model to reason step by step in natural language.
    The explicit 'box your answer' instruction improves answer extraction
    and mirrors standard evaluation practice.
    """
    return (
        "Solve the following math problem step by step.\n"
        "Show all your reasoning clearly.\n"
        "At the end, write your final answer as: Final Answer: <number>\n\n"
        f"Problem: {question}\n\n"
        "Solution:"
    )


def pot_prompt(question: str) -> str:
    """
    Program-of-Thought: ask the model to write executable Python.
    The model offloads arithmetic to the interpreter, so entropy
    should be lower at numeric tokens — this is a useful contrast
    to CoT entropy patterns.
    """
    return (
        "Write a Python program to solve the following math problem.\n"
        "Store the final answer in a variable called `answer`.\n"
        "Only output the code, no explanation.\n\n"
        f"Problem: {question}\n\n"
        "```python\n"
    )


def direct_prompt(question: str) -> str:
    """
    Direct: no reasoning scaffold.
    Used as a baseline to verify that reasoning scaffolds help.
    """
    return (
        f"Problem: {question}\n\n"
        "Answer (number only):"
    )