# -------------------------------------------------------------------------------------------------------------------------------------------
'''
Optimization of RAG (Retrieval-Augmented Generation) in DSPy involves tuning the parameters of the RAG pipeline to improve performance metrics
like accuracy. This includes selecting effective prompts, adjusting LM weights, and incorporating good demonstrations within the prompts. 
The process uses DSPy optimizers, such as BootstrapFewShot, to systematically enhance the quality and cost-efficiency of the RAG program.
'''
#----------------------------------------------------------------------------------------------------------------------------------------------
from typing import List, Union, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_openai.embeddings import OpenAIEmbeddings
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dspy
from dspy.predict.langchain import LangChainModule, LangChainPredict
from dspy import Example # type: ignore
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate.evaluate import Evaluate
import tqdm
import uuid

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# we are making lots of calls to llm, then we set a global cach
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path="cache.db"))

#-----------------------------------------------------------------------------
#           Initialize Qdrant client
#-----------------------------------------------------------------------------
try:
    client = QdrantClient(url="http://localhost:6333/")
    print("Successfully connected to Qdrant.")
except ResponseHandlingException as e:
    print(f"Error connecting to Qdrant: {e}")
    exit(1)


#-----------------------------------------------------------------------------
#           Create Qdrant collection
#-----------------------------------------------------------------------------
collection_name = "MentalHealthCounseling"
embedding_dimension = 1536  # Replace with the actual dimension of your embeddings

# Check if collection exists and delete it if it does
try:
    client.delete_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' deleted successfully.")
except ResponseHandlingException as e:
    print(f"Error deleting collection: {e}")

# Create Qdrant collection
try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )
    print(f"Collection '{collection_name}' created successfully.")
except ResponseHandlingException as e:
    print(f"Error creating collection in Qdrant: {e}")
    exit(1)


#-----------------------------------------------------------------------------
#           LOAD SPLIT EMBEDD Documents
#-----------------------------------------------------------------------------
docs = ArxivLoader(
    # query='"mental health counseling" AND (data OR analytics OR "machine learning")',
    #query='"mental health counseling" AND ("prompt engineering" OR "language models" OR "AI optimization" OR "DSPY")',
    query='Advancement in mental health issues and counseling',
    load_max_docs=5,
    sort_by="submittedDate",
    sort_order="descending",
).load()

print(f"Loaded {len(docs)} documents.")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(docs)

# Embed documents
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


#-----------------------------------------------------------------------------
#           Add Documnets to Qdrant
#-----------------------------------------------------------------------------
for i, doc in enumerate(split_documents):
    vector = embeddings.embed_query(doc.page_content)
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": str(uuid.uuid4()),  # Generate a valid UUID for the point ID
                    "vector": vector,
                    "payload": {"page_content": doc.page_content},
                }
            ],
        )
        print(f"Document {i} added successfully.")
    except ResponseHandlingException as e:
        print(f"Error adding document to Qdrant: {e}")

#-----------------------------------------------------------------------------
#           My Custom RMClient
#-----------------------------------------------------------------------------
class CustomRMClient(dspy.Retrieve):
    def __init__(self, collection_name, client, embeddings):
        self.collection_name = collection_name
        self.client = client
        self.embeddings = embeddings

    def custom_retrieval(self, question, top_k=10):
        #-----------------------------------------------------------------------------
        #        0.0 Handle INPUT Format
        #-----------------------------------------------------------------------------
        try:
            if isinstance(question, dict):
            # Assuming the actual question text is stored under a key, adjust as necessary
                question = question.get('text', '')  
            # Embed the question to get the query vector
            query_vector = self.embeddings.embed_query(question)
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            # Return the result in the expected format
            return [{"long_text": hit.payload["page_content"]} for hit in search_result]
        except ResponseHandlingException as e:
            print(f"Error during retrieval: {e}")
            return []

    def __call__(self, query, k=3):        
        retrieved_documents = self.custom_retrieval(query, top_k=k)
        prediction = dspy.Prediction(passages=retrieved_documents)
        # print('retrieved_documents:', retrieved_documents)
        # print('prediction:', prediction.passages)        
        return prediction



# Instantiate my custom RM client with the embeddings
custom_rm_client = CustomRMClient(collection_name=collection_name, client=client, embeddings=embeddings)

#-----------------------------------------------------------------------------------------------------------------
# https://github.com/stanfordnlp/dspy/blob/main/examples/integrations/qdrant/qdrant_retriever_example.ipynb
# This notebook assumes, you have a Qdrant instance running at http://localhost:6333/. To learn more about setting up Qdrant, you can refer to the quickstart guide.
# from dsp.modules.sentence_vectorizer import OpenAIVectorizer
# from dspy.retrieve.qdrant_rm import QdrantRM
# vectorizer = OpenAIVectorizer(model="text-embedding-3-small")
# qdrant_retriever = QdrantRM(
#     qdrant_client=client,
#     qdrant_collection_name=collection_name,
#     vectorizer=vectorizer,
#     document_field="text",
# )
# dspy.settings.configure(rm=qdrant_retriever)


#-----------------------------------------------------------------------------
#           DSPY INTEGRATION LM RM CONFIGURATION
#-----------------------------------------------------------------------------

# The prediction for the input question in DSPy is done by the Language Model (LM). 
# The Retrieval Model (RM) is used to fetch relevant context or passages that the LM then can use to generate a more accurate response.
# The RM and LM work together to produce the final output/prediction. 


# Configure the LM: the judge
gpt4T = dspy.OpenAI(model="gpt-4-turbo", max_tokens=1000, model_type="chat")

# Configure the RM client in DSPY settings for RM and configure the LM
dspy.settings.configure(lm=gpt4T, rm=custom_rm_client)

#-----------------------------------------------------------------------------
#           RAG SIGNATURE
#-----------------------------------------------------------------------------

# Define the signature for generating answers
# class GenerateAnswer(dspy.Signature):
#     """Answer questions with short factoid answers."""
#     context = dspy.InputField(desc="may contain relevant facts")
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 1 and 5 words")

# # class RAG(dspy.Module):
# #     def __init__(self, num_passages=3):
# #         super().__init__()
# #         self.retrieve = dspy.Retrieve(k=num_passages)
# #         self.generate = dspy.ChainOfThought(GenerateAnswer)

# #     def forward(self, question):
# #         retrieved_context = self.retrieve(question).passages
# #         prediction = self.generate(context=retrieved_context, question=question)
# #         return dspy.Prediction(context=retrieved_context, answer=prediction.answer)
# class RAG(dspy.Module):
#     def __init__(self, num_passages=3):
#         super().__init__()
#         self.retrieve = dspy.Retrieve(k=num_passages)
#         self.generate = dspy.ChainOfThought(GenerateAnswer)

#     def forward(self, question):
#         retrieved_context = self.retrieve(question).passages
#         print()
#         print("retrieved_text:",retrieved_context)
#         print()
#         # context = " ".join([doc['long_text'] for doc in retrieved_context])  # Properly extracting 'long_text'
#         prediction = self.generate(context=retrieved_context, question=question)
#         return dspy.Prediction(context=retrieved_context, answer=prediction.answer)





#-----------------------------------------------------------------------------
#           ZEROSHOTCHAIN AND DSPY INTEGRATION
#-----------------------------------------------------------------------------

# Instantiate the language model
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Define the prompt template
prompt = PromptTemplate.from_template(
    "Given {context}, answer the question `{question}` as a tweet. Your response should only contain the tweet."
)
num_passages=3

# Define the retrieval module
retriever=dspy.Retrieve(k=num_passages)
# retriever = lambda x: dspy.Retrieve(k=num_passages).passages

#-----------------------------------------------------------------------------
#           Helper Function: 
#           TO TRANSFORM THE RETRIEVED DOCUMENTS  TO THE FORMAT DSPY UNDERSTANDS
#-----------------------------------------------------------------------------
# To ensure MY RunnablePassthrough.assign(context=retrieve) works correctly with MY custom RM client, 
# I need to make sure that the retrieve function returns the context in the expected format. HERE IS HOW
# TO Integrate MY custom RM client with the RunnablePassthrough:

#-----------------------------------------------------------------------------
#        0.1 Handle Output Format
#-----------------------------------------------------------------------------
def retrieve(query):
    custom_rm_client = CustomRMClient(collection_name="MentalHealthCounseling", client=client, embeddings=embeddings)
    prediction = custom_rm_client(query)
    context = [doc["long_text"] for doc in prediction.passages]
    # print(context)  # Debugging: Print the context to verify its structure
    return context

# Define the zero-shot chain
zeroshot_chain = (
    RunnablePassthrough.assign(context=retrieve)
    # RunnablePassthrough.assign(context=lambda retriver :[doc for doc in retriver])
    | LangChainPredict(prompt, llm)
    | StrOutputParser()
)
#----------------------------------------------------------------------------------------------------------------------
# Note in DSPY llm call changes to llm predictor that handles the modification of Prompt (moving Signature and change it)
# Prompt for Langchain is the Signature for predictor that tries to modify that during optimization in DSPY
#-------------------------------------------------------------------------------------------------------------------------


# Wrap the chain in a LangChainModule
zeroshot_chain = LangChainModule(
    zeroshot_chain
)

#----------------------------------------------------------------------------------
# 1. Define the Metric
#----------------------------------------------------------------------------------

# Define the assessment signature
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""
    context = dspy.InputField(desc="ignore if N/A")
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")


# Define the assessment composite metric
METRIC = None
def composit_metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output

    context = retrieve(question)
    

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    dope = f"Is the assessed text dope, lit, cool, fire?"
    responsible = "Does the assessed text adhere to responsible AI principles? Consider fairness, transparency, privacy, and ethical implications."
    correct = (
        f"The text above should answer `{question}`. The gold answer is `{answer}`."
    )
    correct = f"{correct} does the assessed text communicate the same idea as the gold answer?"

    with dspy.context(lm=gpt4T):
        faithful = dspy.Predict(Assess)(
            context=context, assessed_text=tweet, assessment_question=faithful
        )
        engaging = dspy.Predict(Assess)(
            context="N/A", assessed_text=tweet, assessment_question=engaging
        )
        dope = dspy.Predict(Assess)(
            context="N/A", assessed_text=tweet, assessment_question=dope
        )
        responsible_ai = dspy.Predict(Assess)(
            context="N/A", assessed_text=tweet, assessment_question=responsible
        )
        correct = dspy.Predict(Assess)(
            context="N/A", assessed_text=tweet, assessment_question=correct
        )

    correct, engaging, dope, responsible_ai, faithful = [
        m.assessment_answer.split()[0].lower() == "yes"
        for m in [correct, engaging, dope, responsible_ai, faithful]
    ]
    score = (faithful + engaging + responsible_ai + dope) if correct and (len(tweet) <= 280) else 0
    # penalize if not correct or longer than it should be.
    if METRIC is not None:
        if METRIC == "correct":
            return correct
        if METRIC == "engaging":
            return engaging
        if METRIC == "faithful":
            return faithful
        if METRIC == "dope":
            return dope
        if METRIC == "responsible_ai":
            return responsible_ai

    if trace is not None:
        return score >= 4
    return score / 4.0

#----------------------------------------------------------------------------------
# 2. SDG
#----------------------------------------------------------------------------------

# Example SDG creation
NUM_SAMPLES_TO_GENERATE = 250#100
question_list = []
answer_list = []

question_llm = ChatOpenAI(model="gpt-4o-mini")
question_prompt = PromptTemplate.from_template(
    "Given a context, generate a question that could be answered by that context. You must only respond with the question. Context:\n{context}\n\Question:\n"
)

question_chain = question_prompt | question_llm | StrOutputParser()

answer_llm = ChatOpenAI(model="gpt-4o")
answer_prompt = PromptTemplate.from_template(
    "Given a context and a question, create a tweet about the question and the context. You must only respond with the tweet. Context:\n{context}\n\n\Question:\n{question}\nTweet:\n"
)

answer_chain = answer_prompt | answer_llm | StrOutputParser()

if NUM_SAMPLES_TO_GENERATE > len(split_documents):
    NUM_SAMPLES_TO_GENERATE = len(split_documents)
    print(f" warning!!! reducing number of samples to {NUM_SAMPLES_TO_GENERATE}")

for context in tqdm.tqdm(split_documents[:NUM_SAMPLES_TO_GENERATE]):
    question = question_chain.invoke({'context': context.page_content})
    answer = answer_chain.invoke({'context': context.page_content, 'question': question})

    question_list.append(question)
    answer_list.append(answer)

train_samples = int(NUM_SAMPLES_TO_GENERATE * 0.8)
dev_samples = int(NUM_SAMPLES_TO_GENERATE * 0.1)
val_samples = int(NUM_SAMPLES_TO_GENERATE * 0.1)

train_set = []
dev_set = []
val_set = []

sample_count = 0

for question, answer in zip(question_list, answer_list):
    if sample_count < train_samples:
        train_set.append(Example(question=question, answer=answer).with_inputs("question"))
    elif sample_count < train_samples + dev_samples:
        dev_set.append(Example(question=question, answer=answer).with_inputs("question"))
    else:
        val_set.append(Example(question=question, answer=answer).with_inputs("question"))

    sample_count += 1


#----------------------------------------------------------------------------------
# 3. Evaluate SET UP (with custom_rm providing context)
#----------------------------------------------------------------------------------

evaluate = Evaluate(
    metric=composit_metric, devset=dev_set, num_threads=8, display_progress=True, display_table=True
)

print()
print("EVALUATE instantiated successfully.")
print()
no_shot_chain=evaluate(zeroshot_chain)
#----------------------------------------------------------------------------------
# 3.1 Evaluate the Zero-Shot Chain
#----------------------------------------------------------------------------------
print()
print('EvaluateThe Zero Shot Chain: ')
print(no_shot_chain)
print()


#----------------
# from dspy.teleprompt import BootstrapFewShot

# # Define the optimizer configuration
# config = {
#     'metric': composit_metric,
#     'max_bootstrapped_demos': 4,
#     'max_labeled_demos': 16,
#     'max_rounds': 1,
#     'max_errors': 5
# }

# # Initialize the optimizer
# fewshot_optimizer = BootstrapFewShot(**config)
# # Assuming you have a trainset defined
# compiled_rag = fewshot_optimizer.compile(student=RAG(), trainset=train_set)


#----------------------------------------------------------------------------------
# 3.2 Optimize the Prompt using BootstrapFewShotWithRandomSearch (with custom_rm providing context)
#----------------------------------------------------------------------------------

teleprompter_optimizer = BootstrapFewShotWithRandomSearch(
    metric=composit_metric,
    max_bootstrapped_demos=5,
    num_candidate_programs=3    
)

#----------------------------------------------------------------------------------
# 3.2 Use the optimized prompt in the RAG model with dspy teleprompter
#----------------------------------------------------------------------------------

# optimized_prompt = teleprompter_optimizer.optimize_prompt(
#     model=query_gpt4,
#     retrieval_function=lambda: custom_rm(train_set),
#     num_iterations=10,
#     num_samples=5
# )

#----------------------------------------------------------------------------------
# 3.3       Use the optimized prompt in the Chain/RAG model with dspy teleprompter:
#           Compile the RAG/Chain program using the optimized prompt
#           Evaluate the Optimized Chain

#----------------------------------------------------------------------------------

optimized_rag = teleprompter_optimizer.compile(zeroshot_chain, trainset=train_set, valset=val_set)
optim_shot_chain= evaluate(optimized_rag)
print(optim_shot_chain)

optimized_prompt_used, output = dspy.settings.langchain_history[-1]
print('Optimized prompt used: ', optimized_prompt_used)
# The compiled RAG now uses the optimized prompt when retrieving and generating answers.





#docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
#docker stop qdrant
# docker rm qdrant
# docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
# check the qdrant container status: docker ps

#--------------------------------------------------------------------------------
#           3. Initialize and Compile the Optimizer
#--------------------------------------------------------------------------------

# # Use the updated prompt from the optimized chain for later queries
# optimized_prompt = optimized_chain.optimized_prompt
# print("Optimized Prompt:", optimized_prompt)
#-----------------------------------------------------------------------------------------
# # Function to use the optimized prompt in subsequent queries
def query_with_optimized_prompt(question):
    return gpt4T(optimized_prompt_used.format(question=question, context=retrieve(question),  tweet_response=""))
import random

for i in random.sample(range(0, val_samples), val_samples):

    question=dev_set[i]['question']
    print()
    print()
    print('question: ', question)
    print()
    prediction=zeroshot_chain(question=question).tweet_response
    print('tweet by naiive prompt: ', prediction[prediction.find('Tweet Response') + len('Tweet Response: '):])
    print()
    print('my tweet response by DSPY Optimized prompt engineering: ', query_with_optimized_prompt(question)[0])
    print()
    print('Groud Thruth Answer: ', dev_set[i]['answer'])
    print()
#-----------------------------------------------------------------------------------------
#  with HF finetunned
#-----------------------------------------------------------------------------------------
# from transformers import pipeline

# # Load your fine-tuned HF model
# hf_model = pipeline('text-generation', model='path/to/your/fine-tuned-model')

# # Define a function to evaluate the HF model
# def evaluate_hf_model(model, dataset, metric):
#     results = []
#     for example in dataset:
#         input_text = example['input']
#         expected_output = example['output']
#         generated_output = model(input_text)[0]['generated_text']
#         score = metric(expected_output, generated_output)
#         results.append(score)
#     return sum(results) / len(results)

# # Evaluate the HF model
# hf_model_results = evaluate_hf_model(hf_model, dev_set, composit_metric)
# print('Fine-Tuned HF Model Results:', hf_model_results)

