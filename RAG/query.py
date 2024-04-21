import argparse
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from prompt import SYS_PROMPT

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
{system_prompt}

Use these as references:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    miniLM = "pkshatech/GLuCoSE-base-ja"
    embeddings = HuggingFaceEmbeddings(model_name= miniLM)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    results = db.similarity_search_with_score(query_text, k=10)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text)

    prompt = ChatPromptTemplate.from_messages([("system" , SYS_PROMPT) , ("human" , "[参考:] {context_text} [問合せ:] {query_text}")])
    print(prompt)

    model = ChatOpenAI()
    chain = prompt | model
    response_text = chain.invoke(
        {
            "context_text": context_text,
            "query_text": query_text,
        }
    )

    # formatted_response = f"Response: {response_text}\n"
    print(response_text.content)

    return response_text

if __name__ == "__main__":
    main()