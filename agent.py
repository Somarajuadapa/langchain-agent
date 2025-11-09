"""
LangChain agent with DuckDuckGo search for current information
"""
from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType


class SimpleHFLLM(LLM):
    """Simple Hugging Face LLM wrapper"""
    model_name: str
    api_token: str
    client: Any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, model_name: str, api_token: str):
        super().__init__(model_name=model_name, api_token=api_token)
        object.__setattr__(self, 'client', InferenceClient(model=model_name, token=api_token))
    
    @property
    def _llm_type(self) -> str:
        return "simple_hf"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call Hugging Face API"""
        try:
            # Use chat_completion for instruction models
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024
            )
            # Extract message content from chat response
            if isinstance(response, dict):
                if "choices" in response and len(response["choices"]) > 0:
                    return response["choices"][0]["message"]["content"]
                elif "generated_text" in response:
                    return response["generated_text"]
            return str(response) if response else ""
        except Exception as e:
            raise Exception(f"HF API error: {str(e)}")


def create_agent(api_key, model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """Create agent with DuckDuckGo search for current information"""
    llm = SimpleHFLLM(model_name, api_key)
    tools = [DuckDuckGoSearchRun()]
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        max_iterations=20,
        max_execution_time=120,
        handle_parsing_errors=True,
        early_stopping_method="generate",
    )
    
    return agent


def execute_query(agent, query):
    """Execute query with web search"""
    try:
        result = agent.invoke({"input": query})
        output = result.get("output", "")
        
        # Check if agent stopped early
        if not output or "stopped due to" in str(result).lower():
            # Try to extract any partial response
            if "intermediate_steps" in result:
                steps = result["intermediate_steps"]
                if steps:
                    last_step = steps[-1]
                    if len(last_step) > 1:
                        return f"Partial response: {str(last_step[1])}\n\nNote: Agent may have stopped early. Try rephrasing your question."
            
            return "The agent stopped early. This might be due to a complex query or API limitations. Try asking a simpler, more direct question."
        
        return output
    except Exception as e:
        error_msg = str(e)
        if "iteration" in error_msg.lower() or "time limit" in error_msg.lower():
            return f"The query took too long or required too many steps. Try asking a simpler, more direct question. Error: {error_msg}"
        raise Exception(f"Failed to execute query: {error_msg}")
