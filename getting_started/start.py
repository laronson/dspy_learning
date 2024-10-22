import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

qa = dspy.Predict("question:str -> answer:str")
res = qa(question="What are high memory and low memory on linux?")
print(res)

dspy.inspect_history(n=1)


cot = dspy.ChainOfThought("question -> response")
res = cot(question="should curly braces appear on their own line?")
print(res)
dspy.inspect_history(n=1)
