from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import CSVLoader

product_loader = CSVLoader('./data/ecommerce_products.csv')
product_documents = product_loader.load()
product_text_splitter = CharacterTextSplitter(chunk_size=1024, separator="\n")
product_texts = product_text_splitter.split_documents(product_documents)
product_search = FAISS.from_documents(product_texts, OpenAIEmbeddings())
product_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=product_search, verbose=True)

@tool("FAQ")
def faq(intput: str) -> str:
    """"useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
    return faq_chain.run(intput)

@tool("Recommend Product")
def recommend_product(input: str) -> str:
    """"useful for when you need to search and recommend products and recommend it to the user"""
    return product_chain.run(input)

tools = [
    Tool(
        name = "Search Order",func=search_order, 
        description="useful for when you need to answer questions about customers orders"
    ),
    recommend_product, faq]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
