import os
import requests
import ujson
import dspy
from dspy.evaluate import (
    SemanticF1,
)  # TODO: Understand why we chose to use this metric to evaluate a multi-class clarification model
import torch
import functools
from litellm import embedding as Embed

RAG_FILE_PATH = "./getting_started/rag_data"


def get_data(rag_file_path: str):
    urls = [
        "https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_500.json",
        "https://huggingface.co/datasets/colbertv2/lotte_passages/resolve/main/technology/test_collection.jsonl",
        "https://huggingface.co/dspy/cache/resolve/main/index.pt",
    ]

    for url in urls:
        filename = f"{rag_file_path}/{os.path.basename(url)}"
        remote_size = int(requests.head(url, allow_redirects=True).headers.get("Content-Length", 0))
        local_size = os.path.getsize(filename) if os.path.exists(filename) else 0

        if local_size != remote_size:
            print(f"Downloading '{filename}'...")
            with requests.get(url, stream=True) as r, open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


# TODO: Fix this so it is not the case that if this is not run before trying to execute on any of the other parts of
# this file, our dspy instance will not be configured.  MUST RUN FIRST!
def dspy_setup(rag_file_path: str):
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    with open(f"{rag_file_path}/ragqa_arena_tech_500.json") as f:
        data = [dspy.Example(**d).with_inputs("question") for d in ujson.load(f)]
        trainset, valset, devset, testset = data[:50], data[50:150], data[150:300], data[300:500]

    metric = SemanticF1()
    evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24, display_progress=True, display_table=3)

    return (metric, evaluate, trainset, valset, devset, testset, lm)


# Define the function that creates our retriever function that we will use to make a call to our vector store
def get_retriever(rag_file_path: str):
    with open(f"{rag_file_path}/test_collection.jsonl") as f:
        corpus = [ujson.loads(line) for line in f]

    # load the contents of the .pt (.pytorch file)
    index = torch.load(f"{rag_file_path}/index.pt")
    max_chars = 4000

    # Create a function that leverages the functools library's lru cache decorator to memoize expensive calls to our
    # retriever
    @functools.lru_cache(maxsize=None)
    def search(query, k=5):
        # Create an embedding from the query input and store the result in a tourchpy tensor.  We specify using
        # text-embedding-3-small which is one of openai's newer embedding models
        query_embeddings = torch.tensor(Embed(input=query, model="text-embedding-3-small").data[0]["embedding"])
        # Execute the query between what is stored in our vector store and what we are trying to search for
        topk_scores, topk_indicies = torch.matmul(index, query_embeddings).topk(k)
        # Pull the top values by index from the corpus data using the indexes from the resulting vector store search
        # operation and return the results (truncated if over the max number of allowed chars in a result)
        topK = [dict(score=score.item(), **corpus[idx]) for idx, score in zip(topk_indicies, topk_scores)]
        return [doc["text"[:max_chars]] for doc in topK]

    return search


# Define a new dspy Module that will be used in the process of answering a question.  In the case of this example, our
# process will contain two steps.  First, a query will be sent through our RAG module to retrieve the top-k items
# closest to our query.  The return from our retriever will then get forwarded to a chain-of-thought dspy module (with a
# signature defined as context, question -> answer) as the context.  The chain-of-thought module will act as a
# generation module for our chain and pass a fully defined prompt with the original question and the context retrieved
# from the rag module to our LM.
class RAGModule(dspy.Module):
    def __init__(self, num_docs=5):
        self.num_docs = num_docs
        self.respond = dspy.ChainOfThought("context, question -> answer")
        self.retriever = get_retriever(RAG_FILE_PATH)

    def forward(self, question):
        context = self.retriever(question, k=self.num_docs)
        return self.respond(context=context, question=question)


metric, evaluate, trainset, valset, devset, testset, lm = dspy_setup(RAG_FILE_PATH)
rag = RAGModule()
rag(question="what are high memory and low memory on linux?")
dspy.inspect_history()


def optimize_rag_prompt(metric, trainset, valset):
    """
    If we evaluate the rag pipeline we use before optimization using the SemanticF1 metric defined through dspy, we score
    around 53%.  With dspy we can do better by optimizing the prompts used throughout our entire pipeline.  In this case,
    we only have one step in our chain that calls an LM so we only need to optimize the prompt that is plugged into our
    CoT module.
    Example optimization output can be found here: https://gist.github.com/okhat/d6606e480a94c88180441617342699eb

    The MIPRO optimizer (Multi-instruction proposal optimizer) works in three distinct steps to optimize a pipelines
    prompts:
        1) Bootstrap task demonstration
        2) Propose new instruction candidates
        3) Search over results from steps 1 and 2 using Bayesian Optimization to find the best combination of variables
        to get the best prompt
    """

    tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)
    optimized_rag = tp.compile(
        RAGModule(),
        trainset=trainset,
        valset=valset,
        # If the optimizer decides to use bootstrapped or labeled demonstrations (e.g. adding few-shot demonstrations)
        # in the optimized prompt, only add at most two of them
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        requires_permission_to_run=False,
    )
    return optimized_rag


## CALLING OPTIMIZED RAG
# optimized_rag = optimize_rag_prompt(metric,trainset,valset)
# pred = optimized_rag(question="cmd+tab does not work on hidden or minimized windows")
# print(pred.response)
# evalutate(optimized_rag)


# Get the history of everything that had been ran against the defined lm to get the cost of how much this pipeline has
# cost
def get_cost_of_pipeline_so_far(lm):
    sum([x["cost"] for x in lm.history if x["cost"] is not None])


# Similar to how you can save models in frameworks like tensorflow, you can save an optimized pipeline for later use
# It seems that the save and load functions are extended from the dspy.Module base class so you must call them on an
# instantiated instance of the module
optimized_rag = RAGModule()  # Pretend that this has been instantiated and optimized
optimized_rag.save("saved_optimized_rag.json")
loaded_rag = RAGModule()
loaded_rag.load("saved_optimized_rag.json")  # New instance is now optimized
