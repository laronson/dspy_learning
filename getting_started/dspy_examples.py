"""
In this file, we create a system that interacts with an LM in order to answer questions about tech and programming.  
To do this, we pull 500 questions from stack overflow, turn them into dspy Example objects, and then split them into
training, validation, development, and testing datasets.

Once we have our data to help build the system, we need to select a metric for the prompt optimizer to use in order to 
optimize prompts that produce bad responses.  For this, we use the concept of checking which responses use the 
"key facts" in their answers AND which responses do not refer to "key facts" as compared to their actual answers in
the dataset.
"""

import ujson, dspy
from dspy.evaluate import SemanticF1
from dotenv import load_dotenv

load_dotenv()

# Define the model with a question->response signature
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
cot = dspy.ChainOfThought("question -> response")

# 500 question--answer pairs from the RAG-QA Arena "Tech" dataset.
with open("./getting_started/ragqa_arena_tech_500.json") as f:
    data = ujson.load(f)
# Convert the data to dspy examples (**d is the same as **kwargs I guess)
example_data = [dspy.Example(**d).with_inputs("question") for d in data]


# Split dataset into training and validation sets.  Good to have between 30 and 300 for both sets
# For prompt optimizers, its good to have more validation data than training data
# good to also have dev data to use as you iterate on your system (aka data to develop with) and test data
# to hold onto to test your trained and validated system once building phase is complete
trainset, valset, devset, testset = (
    example_data[:50],
    example_data[50:150],
    example_data[150:300],
    example_data[300:500],
)
metric = SemanticF1()
example = example_data[40]
prediction = cot(**example.inputs())  # putting **before an object in a function signature unpacks that object
score = metric(example, prediction)

print(f"Question: \t {example.question}\n")
print(f"Gold Reponse: \t {example.response}\n")
print(f"Predicted Response: \t {prediction.response}\n")
print(f"Semantic F1 Score: {score:.2f}")


dspy.inspect_history(n=1)


# Define an evaluator that we can re-use.
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24, display_progress=True, display_table=True)

# Evaluate the Chain-of-Thought program.
evaluate(cot)
