import streamlit as st
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import duckdb
import yaml
import glob
import sqlparse


def get_verified_queries_and_embeddings(directory_path, embedding_model):
    verified_queries_yaml_files = glob.glob(os.path.join(directory_path, '*.yaml'))
    verified_queries_dict = {}
    for file in verified_queries_yaml_files:
        with open(file, 'r') as stream:
            try:
                file_name = file[len(directory_path):-5]
                verified_queries_dict[file_name] = yaml.safe_load(stream)
                verified_queries_dict[file_name]['embeddings'] = embedding_model.encode(verified_queries_dict[file_name]['description'])
            except yaml.YAMLError as exc:
                continue
        
    return verified_queries_dict


def get_verified_sql(embedding_model,user_question,verified_queries_dict,minimum_similarity):
    
    # Get embeddings for user question
    prompt_embeddings = embedding_model.encode(user_question)

    # Calculate embedding similarity for verified queries using cosine similarity
    embeddings_list = [data["embeddings"] for prompt, data in verified_queries_dict.items()]
    verified_queries = list(verified_queries_dict.keys())
    similarities = cosine_similarity([prompt_embeddings], embeddings_list)[0]

    # Find the index of the highest similarity
    max_similarity_index = np.argmax(similarities)

    # Retrieve the most similar prompt using the index
    most_similar_prompt = verified_queries[max_similarity_index]
    highest_similarity = similarities[max_similarity_index]
    sql_query = sqlparse.format(verified_queries_dict[most_similar_prompt]['sql'], reindent=True, keyword_case='upper')

    if highest_similarity >= minimum_similarity / 100.0:
        #st.write("Found a verified query:",most_similar_prompt,'(similarity',round(highest_similarity*100,1),'\%)')
        st.markdown(f"Found a verified query: **{most_similar_prompt}** ({round(highest_similarity*100,1)}% similarity)")
        st.markdown("```sql\n" + sql_query + "\n```")
        return sql_query
    else:
        st.markdown(f"Unable to find a verified query to answer your question. Most similar prompt: **{most_similar_prompt}** ({round(highest_similarity*100,1)}% similarity)")
        return None
    


def chat_with_groq(client,prompt,model):
    
    completion = client.chat.completions.create(
    model=model,
    messages=[
      {
            "role": "user",
            "content": prompt
        }
        ]
    )
  
    return completion.choices[0].message.content


def execute_duckdb_query(query):

    original_cwd = os.getcwd()
    os.chdir('data')
    
    try:
        conn = duckdb.connect(database=':memory:', read_only=False)
        query_result = conn.execute(query).fetchdf().reset_index(drop=True)
    finally:
        os.chdir(original_cwd)


    return query_result


def get_summarization(client,user_question,df,model,additional_context):
    prompt = '''
    A user asked the following question pertaining to local database tables:
    
    {user_question}
    
    To answer the question, a dataframe was returned:

    Dataframe:
    {df}

    In a few sentences, summarize the data in the table as it pertains to the original user question. Avoid qualifiers like "based on the data" and do not comment on the structure or metadata of the table itself
    '''.format(user_question = user_question, df = df)

    if additional_context != '':
        prompt += '''\n
        The user has provided this additional context:
        {additional_context}
        '''.format(additional_context=additional_context)

    return chat_with_groq(client,prompt,model)


def main():
    
    groq_api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(
        api_key=groq_api_key,
        base_url=st.secrets["GROQ_BASE_URL"]
    )

    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('groqcloud_darkmode.png')

    st.title("DuckDB Query Retriever")
    multiline_text = """
    Welcome! Ask questions about employee data or purchase details, like "Show the 5 most recent purchases" or "What was the most expensive purchase?". The app matches your question to pre-verified SQL queries for accurate results.
    """

    st.markdown(multiline_text, unsafe_allow_html=True)

    # Add customization sidebar
    st.sidebar.title('Customization')
    additional_context = st.sidebar.text_input('Enter additional summarization context for the LLM here (i.e. write it in spanish):')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096']
    )
    minimum_similarity = st.sidebar.slider('Minimum Similarity:', 1, 100, value=50)

    user_question = st.text_input("Ask a question:",value='How many Teslas were purchased?')

    if user_question:

        verified_queries_dict = get_verified_queries_and_embeddings('verified-queries/', embedding_model)
        verified_sql_query = get_verified_sql(embedding_model,user_question,verified_queries_dict,minimum_similarity)

        # If a verified query is returned, generate the output and summarization
        if verified_sql_query is not None:
            results_df = execute_duckdb_query(verified_sql_query)
            st.markdown(results_df.to_html(index=False), unsafe_allow_html=True)
            summarization = get_summarization(client,user_question,results_df,model,additional_context)
            st.write('')
            st.write(summarization.replace('$','\\$'))

if __name__ == "__main__":
    main()

