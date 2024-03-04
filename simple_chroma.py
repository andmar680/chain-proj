import chromadb
from chromadb.utils import embedding_functions
from pprint import pprint
import csv

# Create a Chroma client
chroma_client = chromadb.Client()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
collection = chroma_client.create_collection('my_collection') # vector database
# chroma_client.delete_collection('my_collection') # delete collection

# Load sample data (a restaurant menu of items)
with open('menu_items.csv') as file:
    lines = csv.reader(file)
    # Store the name of the menu items in this array. In Chroma, a "document" is a string i.e. name, sentence, paragraph, etc.
    documents = []

    # Store the corresponding menu item IDs in this array.
    metadatas = []

    # Each "document" needs a unique ID. This is like the primary key of a relational database. We'll start at 1 and increment from there.
    ids = []
    id = 1

    # Loop thru each line and populate the 3 arrays.
    for i, line in enumerate(lines):
        if i==0:
            # Skip the first row (the column headers)
            continue

        documents.append(line[1])
        metadatas.append({"item_id": line[0]})
        ids.append(str(id))
        id+=1
    # print(documents)
    # print(metadatas)
    # print(ids)

# Add the documents to the collection
collection.add(
    # documents=["This is a document", "This is another document"], # chroma will encode these documents into vectors and embed them into the database
    documents=documents, # chroma will encode these documents into vectors and embed them into the database
    # metadatas=[{"source": "my_source", "page":1}, {"source": "my_source", "page":2}], # helps us to find the document later
    metadatas=metadatas, # helps us to find the document later
    # ids=["id1", "id2"] # 
    ids=ids # 
)

results = collection.query(
    query_texts=["shrimp"], # chroma will encode this document into a vector and search for similar vectors in the database
    n_results=2,
    include=['distances', 'metadatas', "documents"]
)

pprint(results["documents"])