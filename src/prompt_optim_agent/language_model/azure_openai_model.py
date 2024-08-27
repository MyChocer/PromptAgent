from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import time


class AzureOpenAIModel():
    def __init__(
        self,
        model_name: str,
        temperature: float,
        endpoint: str,
        api_version: str,
        **kwargs):
        try:
            token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default")
        
            self.model = AzureOpenAI(
                azure_ad_token_provider=token_provider,
                azure_endpoint=endpoint,
                api_version=api_version
            )

        except Exception as e:
            print(f"Init openai client error: \n{e}")
            raise RuntimeError("Failed to initialize OpenAI client") from e
        
        self.model_name = model_name
        self.temperature = temperature
        
        self.batch_forward_func = self.batch_forward_chatcompletion
        self.generate = self.gpt_chat_completion

        
    
    def batch_forward_chatcompletion(self, batch_prompts):
        """
        Input a batch of prompts to openai chat API and retrieve the answers.
        """
        responses = []
        for prompt in batch_prompts:
            response = self.gpt_chat_completion(prompt=prompt)
            responses.append(response)
        return responses
    
    
    def gpt_chat_completion(self, prompt):
        messages = [{"role": "user", "content": prompt},]
        backoff_time = 1
        while True:
            try:
                return self.model.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    temperature=self.temperature).choices[0].message.content.strip()
            except Exception as e:
                print(e, f' Sleeping {backoff_time} seconds...')
                time.sleep(backoff_time)
                backoff_time *= 1.5

        
    
        
        