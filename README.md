# PP_LLM
This project is based on utilizing two versions of GPT 3.5 that have been fine-tuned to create an ensemble model for a given dataset. The fine-tuned Large Language Models (LLMs) specialize in detecting the name of the dataset and utilizing the metafeatures to identify the most suitable algorithm for the dataset. It serves as a tool to simplify the process of creating a machine learning model with just a few lines of code, where the entire pipeline is constructed using LLM.

There are two main requirements: determining whether the problem is classification or regression, and providing the dataset split into 'X' and 'y'.

![PP_LLM](animated_automl.gif)

## Classification

```python

import openai
import openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ppllm import PP_LLM

openai.api_key = " " # Introduce your OpenAI key (reminder, you can create a Key with a free account, up to €5 budget "21/08/2023", equivalent to approximately running this framework 500 or more with 3 pipelines solutions)

dataset = openml.datasets.get_dataset(40983) # 40983 is Wilt dataset: https://www.openml.org/search?type=data&status=active&id=40983
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

### Setup and Run LLM pipeline - This will be billed to your OpenAI Account!
automl = PP_LLM(
    llm_model="gpt-3.5-turbo", # You can choose "gpt-4" in case you in case you have a paid account
    iterations=4,
    max_total_time = 3600,
    )

automl.fit(X_train, y_train)

# This process is done only once
y_pred = automl.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print(f'LLM Pipeline accuracy {acc}')

```

## Regression

```python
import openai
import openml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from ppllm import PP_LLM


openai.api_key = " " # Introduce your OpenAI key (reminder, you can create a Key with a free account, up to €5 budget "21/08/2023", equivalent to approximately running this framework 500 or more times with 3 pipelines solutions)
type_task = 'regression'
dataset = openml.datasets.get_dataset(41021) # 41021 is Moneyball dataset: https://www.openml.org/search?type=data&status=active&id=41021
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

### Setup and Run LLM pipeline - This will be billed to your OpenAI Account!
automl = PP_LLM(
    llm_model="gpt-3.5-turbo", # You can choose "gpt-4" in case you in case you have a paid account
    iterations=4,
    task=type_task,
    max_total_time=3600
    )

automl.fit(X_train, y_train)

# This process is done only once
y_pred = automl.predict(X_test)
print("LLM Pipeline MSE:", mean_squared_error(y_test, y_pred))

```
