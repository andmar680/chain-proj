pip install azure-ai-ml

# from langchain_openai import ChatOpenAI

from langchain.llms.openai import AzureOpenAI

# from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings

model = AzureOpenAI(
deployment_name="your-deployment-name",
temperature=0,
streaming=True,
)

embeddings = AzureOpenAIEmbeddings(
deployment_name="your-embedding-deployment-name",
)

export AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com"
