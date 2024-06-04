from datasets import load_dataset
from datasets import DatasetDict

# based on config
raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")

# remove this when done debugging
indices = range(0,100)

dataset_dict = {"train": raw_datasets["train_sft"].select(indices),
                "test": raw_datasets["test_sft"].select(indices)}

raw_datasets = DatasetDict(dataset_dict)
print(raw_datasets)
example = raw_datasets["train"][0]
print(example.keys())
messages = example["messages"]
for message in messages:
  role = message["role"]
  content = message["content"]
  print('{0:20}:  {1}'.format(role, content))
  