"""Creates sample PDFs with realistic content for the RAG demos."""
from fpdf import FPDF
from pathlib import Path


def make_pdf(path: str, pages: list[tuple[str, str]]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for title, body in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.multi_cell(0, 10, title)
        pdf.set_font("Helvetica", size=11)
        pdf.ln(3)
        pdf.multi_cell(0, 7, body.strip())
    pdf.output(path)
    print(f"Created {path}  ({len(pages)} pages)")


AI_PAGES = [
    ("Introduction to Machine Learning",
     "Machine learning is a branch of artificial intelligence that enables systems to learn "
     "from data and improve their performance without being explicitly programmed. "
     "There are three main types of machine learning: supervised learning, unsupervised "
     "learning, and reinforcement learning.\n\n"
     "In supervised learning, the model is trained on labelled data. Common examples include "
     "email spam detection, image classification, and house price prediction."),

    ("Neural Networks",
     "A neural network is a computational model loosely inspired by the human brain. "
     "It consists of layers of interconnected nodes called neurons. The first layer "
     "is the input layer, the last is the output layer, and everything in between "
     "are hidden layers.\n\n"
     "Deep learning refers to neural networks with many hidden layers. These deep "
     "networks excel at image recognition, speech recognition, and natural language "
     "processing. GPT and BERT are examples of deep learning models."),

    ("Large Language Models",
     "Large Language Models (LLMs) are neural networks trained on vast amounts of text. "
     "They learn to predict the next word in a sequence, which forces them to build "
     "rich internal representations of language and knowledge.\n\n"
     "GPT-4, Claude, and Gemini are examples of LLMs. They are trained on trillions of "
     "tokens and contain billions of parameters. Fine-tuning allows these general models "
     "to be specialised for specific tasks such as coding or medical advice."),

    ("Retrieval-Augmented Generation",
     "Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model. "
     "Instead of relying solely on the LLM's memorised knowledge, RAG fetches relevant "
     "documents at query time and provides them as context.\n\n"
     "The pipeline has three stages: documents are split into chunks and embedded into a "
     "vector space; when a question arrives the nearest chunks are retrieved; "
     "the chunks and question are sent to the LLM which generates a grounded answer. "
     "RAG reduces hallucination and lets you update the knowledge base without retraining."),

    ("Vector Databases",
     "A vector database stores high-dimensional vectors and supports fast similarity search. "
     "Popular choices include ChromaDB, Pinecone, Weaviate, and Qdrant.\n\n"
     "The most common indexing algorithm is HNSW (Hierarchical Navigable Small World). "
     "HNSW builds a graph where nearby vectors are connected, enabling sub-linear search "
     "time. Cosine similarity and dot product are the most common distance metrics."),
]

PYTHON_PAGES = [
    ("Python Basics",
     "Python is a high-level, general-purpose programming language created by Guido van Rossum "
     "and first released in 1991. It emphasises code readability and uses indentation to define "
     "code blocks rather than curly braces.\n\n"
     "Python supports multiple programming paradigms including procedural, object-oriented, "
     "and functional programming. Its large standard library and active ecosystem of third-party "
     "packages make it popular for web development, data science, and automation."),

    ("Python Data Structures",
     "Python has four built-in collection types: lists, tuples, sets, and dictionaries.\n\n"
     "A list is an ordered, mutable sequence: [1, 2, 3]. "
     "A tuple is an ordered, immutable sequence: (1, 2, 3). "
     "A set is an unordered collection of unique elements: {1, 2, 3}. "
     "A dictionary maps keys to values: {'name': 'Alice', 'age': 30}.\n\n"
     "Lists are the most flexible; tuples are faster and used for fixed data; "
     "dictionaries are the go-to for fast key-based lookup."),

    ("Python Functions and Scope",
     "Functions in Python are defined with the def keyword. They support default arguments, "
     "keyword arguments, and variable-length argument lists via *args and **kwargs.\n\n"
     "Python uses LEGB scope resolution: Local, Enclosing, Global, Built-in. "
     "A variable is looked up in the local scope first, then in any enclosing function "
     "scopes, then in the module global scope, and finally in the built-in namespace."),
]

HISTORY_PAGES = [
    ("The Industrial Revolution",
     "The Industrial Revolution began in Britain in the late 18th century and spread across "
     "Europe and North America during the 19th century. It marked the transition from "
     "agrarian, handicraft economies to manufacturing and industry.\n\n"
     "Key inventions included the steam engine, the spinning jenny, and the power loom. "
     "The railway network expanded rapidly, connecting cities and enabling mass transport "
     "of goods and people for the first time."),

    ("World War I",
     "World War I lasted from 1914 to 1918 and involved most of the world's great powers. "
     "It was triggered by the assassination of Archduke Franz Ferdinand of Austria-Hungary "
     "in Sarajevo on 28 June 1914.\n\n"
     "The war introduced industrial-scale warfare: machine guns, poison gas, tanks, and "
     "aircraft were all used. The conflict resulted in approximately 20 million deaths "
     "and reshaped the political map of Europe."),

    ("The Space Race",
     "The Space Race was a 20th-century competition between the United States and the Soviet "
     "Union for supremacy in spaceflight. It began with the Soviet launch of Sputnik 1 "
     "in October 1957, the first artificial satellite.\n\n"
     "The US landed the first humans on the Moon on 20 July 1969 during the Apollo 11 mission. "
     "Neil Armstrong became the first person to walk on the Moon, followed by Buzz Aldrin. "
     "The Space Race accelerated advances in rocketry, computing, and telecommunications."),
]


if __name__ == "__main__":
    Path("docs").mkdir(exist_ok=True)
    make_pdf("docs/ai_concepts.pdf",   AI_PAGES)
    make_pdf("docs/python_guide.pdf",  PYTHON_PAGES)
    make_pdf("docs/world_history.pdf", HISTORY_PAGES)
