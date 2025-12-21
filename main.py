from dotenv import load_dotenv
from llama_index.llms.sarvam import Sarvam
import os
from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, settings
from llama_index.core.settings import Settings

from llama_index.core.prompts.prompts import SimpleInputPrompt
load_dotenv()
apikey=os.getenv("SARVAM_API_KEY")

# from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.embeddings.fastembed import FastEmbedEmbedding

embed_model = FastEmbedEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

document=SimpleDirectoryReader(r"D:\journey\ai\pdf-qa\data").load_data()


# llm = Sarvam(
# 	context_window=4096,
# 	max_new_tokens=256,
# 	generate_kwargs={"temperature": 0.0, "do_sample": False},
# 	system_prompt=system_prompt,
# 	query_wrapper_prompt=query_wrapper_prompt,
# 	tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1"),
# 	model_name="sarvam-m",
# )

llm = Sarvam(
    api_key=apikey,
    model="sarvam-m",
    context_window=4500,
    max_tokens=512,
)
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024
Settings.system_prompt = """
You are a Q&A assistant.
Answer strictly based on the provided documents. You should complete 10 points even if you've to exceed token limit.
If the answer is not in the documents, say so.
"""
# query_wrapper_prompt = system_prompt
# service_context = ServiceContext.from_defaults(
# 	chunk_size=1024,
# 	llm=llm,
# 	embed_model=embed_model
# )


index = VectorStoreIndex.from_documents(document)
query_engine = index.as_query_engine()
print(Settings.system_prompt)
print(type(Settings.llm))
response = query_engine.query("Can you find limitations of this paper?")
print(response)