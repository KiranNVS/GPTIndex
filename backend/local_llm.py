import subprocess
from typing import List, Optional

from langchain.llms.base import LLM

TIMEOUT_IN_SECONDS = 60
LLM_COMMAND = "./backend/alpaca/chat"
MODEL_PATH = "backend/alpaca/ggml-alpaca-7b-q4.bin"

def run_command(command: str, *args, **kwargs) -> str:
    try:
        completed_process = subprocess.run(
            [command] + list(args),
            capture_output=True, 
            timeout=TIMEOUT_IN_SECONDS,
            **kwargs
        )
    
    except subprocess.TimeoutExpired:
        return "Program timed out"
    
    except Exception as e:
        return f'Fatal error trying to run command: {e}'
    
    output = completed_process.stdout.decode('utf-8')

    if completed_process.returncode != 0:
        output = '\nError:' + completed_process.stderr.decode('utf-8')

    return output


class LocalLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        llm_output = run_command(LLM_COMMAND, "--prompt", prompt, "-m", MODEL_PATH)
        
        return llm_output