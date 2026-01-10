from utils.llama_wrapper import LlamaLangChainWrapper

# Initialize LLaMA LLM
llm_model = LlamaLangChainWrapper()

def generate_llm_response(query: str, similar_items: list, configuration_item: str = ""):
    if not llm_model:
        return "LLM service unavailable."

    context = "\n".join([
        f"Similar {item['source']} {i+1}: {item['training_text']}"
        for i, item in enumerate(similar_items[:3])
    ])

    prompt = f"""
You are an IT support assistant for ServiceNow.

USER ISSUE:
{query}

CONFIGURATION ITEM:
{configuration_item}

SIMILAR CONTEXT (Incidents and KB Articles):
{context}

Your task:
1. Analyze the user's issue and the provided context.
2. Use relevant KB articles and past incident resolutions to propose a solution.
3. Provide a clear, step-by-step resolution (3-6 steps).
4. If escalation is needed, mention the appropriate assignment group.

Return only the solution steps in a concise format.
"""

    # --- Call LLaMA and extract plain text ---
    try:
        llm_response = llm_model.invoke(prompt)  # safer than __call__
        if isinstance(llm_response, list):
            # extract text from first message
            llm_text = llm_response[0].content
        else:
            llm_text = str(llm_response)
        return llm_text.strip()
    except Exception as e:
        return f"LLaMA Error: {e}"


