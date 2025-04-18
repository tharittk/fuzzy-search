import openai
from dotenv import dotenv_values
from azure.identity import get_bearer_token_provider, ClientSecretCredential
from typing import List, Dict, Any, Tuple
import traceback
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_not_exception_type,
)

config = dotenv_values("azure.env")
tenant_id = config["AZURE_TENANT_ID"]
client_id = config["AZURE_CLIENT_ID"]
client_secret = config["AZURE_CLIENT_SECRET"]
endpoint = config["OPENAI_ENDPOINT"]


def _get_bearer_token_provider() -> str:
    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )

    bearer_token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    access_token = bearer_token_provider()
    return access_token


class OpenAIService:
    def __init__(
        self,
        gpt_model_name="gpt-4o",
        gpt_model={
            "resource": "OpenAI-gpt-4o",
            "api_version": "2024-05-01-preview",
        },
    ):
        self.gpt_model_name = gpt_model_name
        self.gpt_model = gpt_model
        self.token_provider = _get_bearer_token_provider

    def create_data(
        self,
        system: str,
        prompt: str,
    ) -> Tuple[Dict[str, Any], int]:
        data = dict()

        messages = []
        if system != "":
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})
        data.update({"messages": messages})

        # TODO:
        data.update({"temperature": 0})
        return data

    @retry(
        wait=wait_random_exponential(min=5, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_not_exception_type(openai.BadRequestError),
    )
    def create_request(
        self, data: Dict[str, Any], stream: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        openai_resource = self.gpt_model["resource"]

        try:
            deployment_client = openai.AzureOpenAI(
                api_version=self.gpt_model["api_version"],
                azure_endpoint=endpoint,
                azure_ad_token_provider=self.token_provider,
            )

            response = deployment_client.chat.completions.create(
                model=openai_resource,
                messages=data["messages"],
                temperature=data["temperature"],
                stream=stream,
                seed=42,
            )
            if not stream:
                ans = response.choices[0].message.content.strip()

            if stream:
                return response
            else:
                return ans
        except Exception as e:
            traceback.print_exc()
            return e


# model = {
#     "gpt-4o": {
#         "resource": "OpenAI-gpt-4o",
#         "api_version": "2024-05-01-preview",
#     }
# }

if __name__ == "__main__":
    model_name = "gpt-4o"
    openai_service = OpenAIService()
    data = openai_service.create_data(
        system="You are the humorous conversational AI. Anything user say, response like you're joking",
        prompt="Where is the file database that have list of oil and gas production per day,",
    )

    response = openai_service.create_request(data)
    print(response)
