"""
slm_phi_client.py
=================

Client for calling the Phi SLM (Small Language Model) to generate natural language explanations using Hugging Face Transformers.

Requirements:
    pip install transformers torch
    pip install --upgrade openai  # Requires openai>=1.0.0

Usage:
    from slm_phi_client import get_phi_explanation
    explanation = get_phi_explanation("Your prompt here")
"""

import openai
import os
import logging
from dotenv import load_dotenv

# Set up logger
logger = logging.getLogger("slm_phi_client")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

load_dotenv('VFLClientModels/.env')
# Set your OpenAI API key (recommended: use environment variable for security)
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or set directly: openai.api_key = "sk-..."

# Updated for openai>=1.0.0

def get_phi_explanation(prompt: str, max_new_tokens: int = 512, credit_score: float = None) -> str:
    """
    Generate a natural language explanation using OpenAI GPT with score-based tone guidance.
    Args:
        prompt (str): The prompt to send to the model.
        max_new_tokens (int): Maximum number of tokens to generate.
        credit_score (float): The customer's credit score for tone adjustment.
    Returns:
        str: The generated explanation.
    """
    logger.info(f"[get_phi_explanation] Prompt received with score {credit_score}:\n{prompt}")
    
    # Determine system prompt based on credit score
    if credit_score is not None:
        if credit_score > 750:
            system_prompt = (
                "You are a credit analysis system that provides factual explanations of credit scores. "
                "This customer has an EXCELLENT credit score (>750). Focus on the positive impact of their strong features. "
                "Provide factual analysis only. Keep explanations brief (2-3 sentences)."
            )
        elif credit_score >= 700:
            system_prompt = (
                "You are a credit analysis system that provides factual explanations of credit scores. "
                "This customer has a GOOD credit score (700-750). Highlight their solid features and note areas for enhancement. "
                "Provide factual analysis only. Keep explanations brief (2-3 sentences)."
            )
        elif credit_score >= 600:
            system_prompt = (
                "You are a credit analysis system that provides factual explanations of credit scores. "
                "This customer has an ABOVE-AVERAGE credit score (600-700). Provide balanced analysis of their features. "
                "Provide factual analysis only. Keep explanations brief (2-3 sentences)."
            )
        elif credit_score >= 400:
            system_prompt = (
                "You are a credit analysis system that provides factual explanations of credit scores. "
                "This customer has a BELOW-AVERAGE credit score (400-600). Focus on critical features requiring improvement. "
                "Provide factual analysis only. Keep explanations brief (2-3 sentences)."
            )
        else:
            system_prompt = (
                "You are a credit analysis system that provides factual explanations of credit scores. "
                "This customer has a VERY POOR credit score (<400). Emphasize critical features significantly impacting their score. "
                "Provide factual analysis only. Keep explanations brief (2-3 sentences)."
            )
    else:
        system_prompt = (
            "You are a credit analysis system that provides factual explanations of credit scores. "
            "Provide factual analysis only. Be concise and clear. Limit your answers to two or three short sentences."
        )
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ],
        max_tokens=max_new_tokens,
        temperature=0.5,
        n=1,
    )
    return response.choices[0].message.content.strip()

# Example usage (uncomment to test)
# if __name__ == "__main__":
#     prompt = "Explain how income and debt affected the credit score."
#     print(get_phi_explanation(prompt, credit_score=720))  # Example with score
#     print(get_phi_explanation(prompt))  # Example without score 