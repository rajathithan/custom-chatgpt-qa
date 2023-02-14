#!/usr/bin/env python
# coding: utf-8

import openai
from openai.embeddings_utils import get_embedding

import pandas as pd
import numpy as np
import pyarrow
import pinecone

from transformers import GPT2TokenizerFast

from sys import getsizeof
from tqdm.auto import tqdm
import json
import time
import sister
from loguru import logger



# # Dataset Preparation
# 
# * Dataset is taken from Kaggle https://www.kaggle.com/datasets/jithinanievarghese/drugs-side-effects-and-medical-condition?resource=download
# * Took only 3 columns out of this dataset (drug_name, medical_condition and side_effects)
# * Renamed the side_effects column as the context column
# * Added the question column based on drug_name and medical_condition
# * Combined all the text data into text column with keys topic, question and answer
# * Used the transformer pre-trained gp2 tokenizer to get the number of tokens (feature vectors) for the given text in the text column

# In[3]:
logger.info("!!!OpenAI-Custom Q&A ChatGPT System!!!")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
df = pd.read_csv('./drugs_side_effects_drugs_com.csv')
df = df[["drug_name","medical_condition", "side_effects"]]
df = df.dropna()
df = df.assign(questions=lambda x: "what is the side effect of taking drug " + x.drug_name + "for the " + x.medical_condition + " medical condition.")
questions = df.pop('questions')
df.insert(2, 'questions', questions)
df.rename(columns = {'side_effects':'context'}, inplace = True)
df['text'] = "Topic: " + df.drug_name + " - " + df.medical_condition + "; Question: " + df.medical_condition + " - " + df.questions + "; Answer: " + df.context
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x,max_length=1024,truncation=True)))
# logger.debug(df['n_tokens'])
# logger.debug(df.head())
logger.debug(df['n_tokens'])
logger.debug(df.head())


# * Checking the histogram for the distribution of tokens in the dataset

# In[67]:


# df.n_tokens.hist(bins=100, log=True)


# * Trim the dataset , remove data above 2000 tokens

# In[68]:


df = df[df.n_tokens < 1024]


# # Get embedding vector list from OpenAI Embedding Model
# * Model selected is "text-search-curie-doc-001"
# * Free tier API rate limit is 60 per minute, so for every 50 rows in the dataframe give a sleep interval of 1 minute.
# * It took 3 hrs and $14 in free tier expense to get the vector list for each row.
# * Finally stored the vector list in Parquet format. 

# In[ ]:


openai.api_key = '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'#from openai
model = 'curie'


# embedding = get_embedding('Get embedding vector list from OpenAI Embedding Model', engine=f'text-search-{model}-doc-001')

#free sister with bert embdding example:
bert_embedding = sister.BertEmbedding(lang="en")
sentences = ["I am a dog.", "I want be a cat."]
vectors = bert_embedding(sentences)
logger.debug("00embedding vectors:",vectors)
count = 0
embed_array = []
for index, row in df.iterrows():
    count += 1
    logger.debug("bert_embedding text:"+row['text'])
    logger.debug("bert_embeded:",bert_embedding)
    bert_embedding = bert_embedding(row['text'])
    logger.debug("bert_embeded:",bert_embedding)
    embed_array.append(bert_embedding)
    if count == 5:
        logger.debug("embed_array:",embed_array)
        time.sleep(62)
        count = 0 
    
df.insert(6, "embeddings", embed_array, True)       
df.to_parquet('/Users/yangboz/git/yangboz/custom-chatgpt-qa/curie_embeddings.parquet')
logger.debug(df.head())


# In[77]:


df1 = pd.read_parquet('./curie_embeddings.parquet')
df1.head()


# # Creation of Pinecone Index for embedding Vectors list
# * Pincode DB cannot have metadata size greater than 5 KB, so we will assign a unique id for each row and use that mapping for our text data
# * Create Pinecode Index after initializing the pinecone client with API key and environment, we will use cosine similarity metric for our vector embeddings.
# * Update the Index with the vectors list.
# * Create a metadata mapping with id and text columns to see if we are able to retrieve the context from the pinecone vector DB

# In[78]:


too_big = []

for text in df['text'].tolist():
    if getsizeof(text) > 1024:
        too_big.append((text, getsizeof(text)))

logger.debug(f"{len(too_big)} / {len(df)} records are too big")


# In[79]:


df['id'] = [str(i) for i in range(len(df))]
df.head()


# In[81]:


pinecone.init(
    api_key='d2937528-4cf8-426a-bb13-73771763192d',#from https://app.pinecone.io/
    environment='us-west1-gcp'
)

index_name = 'chatgpt-demo'

if not index_name in pinecone.list_indexes():
    pinecone.create_index(
        index_name, dimension=len(df['embeddings'].tolist()[0]),
        metric='cosine'
    )

index = pinecone.Index(index_name)


# In[83]:


batch_size = 32

for i in tqdm(range(0, len(df), batch_size)):
    i_end = min(i+batch_size, len(df))
    df_slice = df.iloc[i:i_end]
    to_upsert = [
        (
            row['id'],
            row['embeddings'],
            {
                'drug_name': row['drug_name'],
                'medical_condition': row['medical_condition'],
                'n_tokens': row['n_tokens']
            }
        ) for _, row in df_slice.iterrows()
    ]
    index.upsert(vectors=to_upsert)


# In[84]:


mappings = {row['id']: row['text'] for _, row in df[['id', 'text']].iterrows()}


# In[85]:


with open('./mapping.json', 'w') as fp:
    json.dump(mappings, fp)


# # Test the Question & Answer system
# * Load the Pinecode Index
# * Test the context retrieval from the context embeddings in Pinecone
# * Build the query, encode it, retrieve the context after passing on to the OpenAI generative model "text-davinci-002"
# * Add the required instructions, so the model generates the output based on give instructions. 

# In[86]:


def load_index():
    pinecone.init(
        api_key='d2937528-4cf8-426a-bb13-73771763192d',  # app.pinecone.io
        environment='us-west1-gcp'
    )

    index_name = 'hnxbdqa'

    if not index_name in pinecone.list_indexes():
        raise KeyError(f"Index '{index_name}' does not exist.")

    return pinecone.Index(index_name)


# In[87]:


index = load_index()


# In[88]:


def create_context(question, index, max_len=3750, size="curie"):
    """
    Find most relevant context for a question via Pinecone search
    """
    q_embed = get_embedding(question, engine=f'text-search-{size}-query-001')
    res = index.query(q_embed, top_k=5, include_metadata=True)
    

    cur_len = 0
    contexts = []

    for row in res['matches']:
        text = mappings[row['id']]
        cur_len += row['metadata']['n_tokens'] + 4
        if cur_len < max_len:
            contexts.append(text)
        else:
            cur_len -= row['metadata']['n_tokens'] + 4
            if max_len - cur_len < 200:
                break
    return "\n\n###\n\n".join(contexts)


# In[89]:


create_context("what is the side effect of taking drug doxycycline for Acne medical condition", index)


# In[98]:


def answer_question(
    index=index,
    fine_tuned_qa_model="text-davinci-002",
    question="Do i get any side effect if I take ibuprofen?",
    instruction="Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nAnswer:",
    max_len=1024,
    size="curie",
    debug=False,
    max_tokens=400,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        index,
        max_len=max_len,
        size=size,
    )
    if debug:
        logger.debug("Context:\n" + context)
        logger.debug("\n\n")
    try:
        # fine-tuned models requires model parameter, whereas other models require engine parameter
        model_param = (
            {"model": fine_tuned_qa_model}
            if ":" in fine_tuned_qa_model
            and fine_tuned_qa_model.split(":")[1].startswith("ft")
            else {"engine": fine_tuned_qa_model}
        )
        #logger.debug(instruction.format(context, question))
        response = openai.Completion.create(
            prompt=instruction.format(context, question),
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            **model_param,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        logger.debug(e)
        return ""


# In[92]:


instructions = {
    "conservative Q&A": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nAnswer:",
    "paragraph about a question":"Write a paragraph, addressing the question, and use the text below to obtain relevant information\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nParagraph long Answer:",
    "bullet point": "Write a bullet point list of possible answers, addressing the question, and use the text below to obtain relevant information\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nBullet point Answer:",
    "summarize problems given a topic": "Write a summary of the problems addressed by the questions below\"\n\n{0}\n\n---\n\n",
    "just instruction": "{1} given the common questions and answers below \n\n{0}\n\n---\n\n",
    "summarize": "Write an elaborate, paragraph long summary about \"{1}\" given the questions and answers from a public forum on this topic\n\n{0}\n\n---\n\nSummary:",
}


# In[93]:


logger.debug(answer_question(index, question="are there side effects for paracetamol drug", 
                            instruction = instructions["summarize problems given a topic"], debug=True))


# In[95]:


logger.debug(answer_question(index, question="are there side effects for paracetamol drug", 
                            instruction = instructions["conservative Q&A"], debug=False))


# In[99]:


logger.debug(answer_question())


# In[100]:


logger.debug(answer_question(index, question="I am planning to take cefixime drug, do i get any side effect", 
                            instruction = instructions["summarize"], debug=False))


# In[102]:


logger.debug(answer_question(index, question="I am taking metoclopramide drug, what are the side effects", 
                            instruction = instructions["bullet point"], debug=False))

