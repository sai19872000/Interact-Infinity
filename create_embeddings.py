# this program creates embeddings for the given pdf file
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# create a function to create embeddings for the given pdf file
def create_embeddings_from_pdf(file="EBZ_22_1251001_01_Cardmember_Agreement_Prime_093022_4.0.pdf",
                               chunk_size=1024,
                               chunk_overlap=64,
                               model_name="roberta-base",
                               embedding_function=HuggingFaceEmbeddings,
                               persist_directory="faiss_embeddings_discover_credit_card_cfpb"):
    

    # load the pdf file

    loader = PyPDFLoader(file)
    documents = loader.load_and_split()
    #print(len(documents))
    #print(documents[0].page_content)

    # chunk the text into smaller pieces with overlap

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"length of document chunks = $len(texts))")

    # create the embeddings

    embeddings = embedding_function(
        #model_name="sentence-transformers/all-MiniLM-L6-v2"
        model_name=model_name
    )

    # save embeddings into a chroma database for fast retrieval. Only needs to be done once
    #db = Chroma.from_documents(texts, embeddings, persist_directory="db_roberta")

    # load the database from disk.
    # db = Chroma(persist_directory="db_roberta", embedding_function=embeddings)

    # create FAISS index for fast retrieval

    faissIndex = FAISS.from_documents(texts, embeddings)
    faissIndex.save_local(persist_directory)

# call the function
create_embeddings_from_pdf(file="Discover_Cardmember_Agreement_Page1.pdf",persist_directory="faiss_dcmap1")