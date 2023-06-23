#program to create a chatbot using gpt4-all
# adapted from https://www.mlexpert.io/prompt-engineering/private-gpt4all
#import all necessay langchain modules
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from pdf2image import convert_from_path

# load the pdf file
# loader = PyPDFLoader("ms-financial-statement.pdf")
loader = PyPDFLoader("/home/sai/discover_company/Interact-Infinity/EBZ_22_1251001_01_Cardmember_Agreement_Prime_093022_4.0.pdf")
documents = loader.load_and_split()
print(len(documents))

print(documents[0].page_content)

# chunk the text into smaller pieces with overlap

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64
)
texts = text_splitter.split_documents(documents)
print(len(texts))

# create the embeddings

embeddings = HuggingFaceEmbeddings(
    #model_name="sentence-transformers/all-MiniLM-L6-v2"
    model_name="roberta-base"
)

# save embeddings into a chroma database for fast retrieval. Only needs to be done once
#db = Chroma.from_documents(texts, embeddings, persist_directory="db_roberta")

# load the database from disk.
# db = Chroma(persist_directory="db_roberta", embedding_function=embeddings)

# create FAISS index for fast retrieval

# faissIndex = FAISS.from_documents(texts, embeddings)
# faissIndex.save_local("faiss_embeddings_discover_credit_card_cfpb")
db_faiss = FAISS.load_local("faiss_embeddings_discover_credit_card_cfpb", embeddings)

# create chain. load gpt4all model

# llm = GPT4All(
#     model="./ggml-gpt4all-j-v1.3-groovy.bin",
#     n_ctx=1000,
#     backend="gptj",
#     verbose=False
# )

#load Nous-Hermes model in gpt4all
llm = GPT4All(
    model="./nous-hermes-13b.ggmlv3.q4_0.bin",
    n_ctx=1000,
    backend="nous-hermes",
    verbose=False
)


# create the retrieval chain

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_faiss.as_retriever(search_type="similarity",search_kwargs={"k":10}),
    #retriever=db.as_retriever(search_kwargs={"k":3}),
    return_source_documents=True,
    verbose=False
)

# ask a question

res = qa(f"""
    What are the interest rates for the credit card?
    Extract it from the text.
""")

# res = qa(f"""
#     How much is the dividend per share during during 2022?
#     Extract it from the text.
# """)


# res = qa(f"""
#          what is the admistrative proceeding?
#             Extract it from the text.
#             """)

print(res["result"])