import datetime
import csv
import time
import openai
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code
from .similarity_optimal_transport import TransferedPipelines

list_pipelines = []

def possible_errors(e):
    if isinstance(e, str):
        consideration = ''
        if 'Cannot use mean strategy' in e or 'could not convert string to float' in e or 'non-numeric data' in e:
            consideration = """Consider calling the package 'ColumnTransformer', that is, first identify automatically all the categorical column names, automatically numeric column names and then add this step to the Pipeline 'pipe' with the original column names. e.g. the columns name can be detect as:  " \
            categorical_columns_dataset = X_train.select_dtypes(include=['object']).columns
            numeric_columns_dataset = X_train.select_dtypes(include=[np.number]).columns
                            """
        if 'missing values' in e or 'contains NaN' in e:
            consideration = 'Consider calling SimpleImputer to replace Nan values with the mean'
        if 'No module named' in e or 'cannot import' in e or 'is not defined' in e:
            consideration = 'Consider the error to see which package you did not call to create the Pipeline and add it this time'
        return consideration

def get_prompt(
        X=None, y=None, task='classification', **kwargs
):
    if task == 'classification':
        metric_prompt = 'Log loss'
        additional_data = ''
        additional_instruction_code = " "
    else:
        metric_prompt = 'Mean Squared Error'
        additional_data = ''
        additional_instruction_code = ""
    similar_pipelines = TransferedPipelines(X_train=X, y_train=y, task=task, number_of_pipelines=3)
    return f"""
The dataframe ‘X_train’ is loaded in memory.
This code was written by an expert data scientist working to create a suitable preprocessing pipeline for a dataframe. It is a snippet of code that imports the packages necessary to create a ‘sklearn’ preprocesing pipeline. This code takes inspiration from previous similar pipelines and their respective ‘{metric_prompt}’ which worked for a related dataframe. Those examples contain the word ‘data’, which refers to ‘X_train’. The previous similar pipelines for this dataframe are:
“
{similar_pipelines}
“
Your work is to create a pipeline for preprocessing (without estimator):

Code formatting for each pipeline created:
```python
(You MUST import all the libraries you will need (e.g., numpy, make_pipeline, SimpleImputer, etc.) to create a preprocessing pipeline object called 'preprocessing_pipe'. In addition, call its respective 'fit_transform' function to feed the model with 'X_train' (transformed_df = preprocessing_pipe.fit_transform(X_train))
In addition, always called  SimpleImputer(strategy="median") at the END of the pipeline.
{additional_instruction_code}
```end

Each codeblock generates exactly one useful pipeline. Which will be evaluated with "{metric_prompt}". 
Each codeblock ends with "```end" and starts with "```python"
{additional_data}
Codeblock:
""", similar_pipelines

# Each codeblock either generates {how_many} or drops bad columns (Feature selection).


def build_prompt_from_df(X=None, y=None,
        task='classification'):

    prompt, similar_pipelines = get_prompt(
        X=X,
        y=y,
        task=task,
    )

    return prompt, similar_pipelines


def generate_dataset(
        X,
        y,
        model="gpt-3.5-turbo",
        just_print_prompt=False,
        iterative=1,
        display_method="markdown",
        task='classification',
        identifier = ""
):
    global list_pipelines # To make it available to sklearn_wrapper in case the time out is reached
    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print
    prompt, similar_pipelines = build_prompt_from_df(
        X=X,
        y=y,
        task=task,
    )

    if just_print_prompt:
        code, prompt = None, prompt
        return code, prompt, None

    def generate_code(messages):
        if model == "skip":
            return ""

        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=500,
        )
        code = completion["choices"][0]["message"]["content"]
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    def execute_and_evaluate_code_block(code):
        if task == "classification":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,  stratify=y, random_state=0)
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(random_state=0)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor(random_state=0)
        try:
            transformed_df = run_llm_code(
                code,
                X_train,
                y_train,
            )
            model.fit(transformed_df, y_train)
            performance = model.score(X_test, y_test)
        except Exception as e:
            model = None
            display_method(f"Error in code execution. {type(e)} {e}")
            display_method(f"```python\n{format_for_display(code)}\n```\n")
            return e, None, None

        return None, performance, model

    messages = [
        {
            "role": "system",
            "content": "You are an expert datascientist assistant creating a preprocessing Pipeline for a dataset X_train. You answer only by generating code. Let’s think step by step",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    display_method(f"*Dataset with specific description, task:*\n {task}")
    list_codeblocks = []
    list_performance = []
    n_iter = iterative
    i = 0
    while i < n_iter:
        try:
            code = generate_code(messages)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            time.sleep(60)  # Wait 1 minute before next request
            continue
        i = i + 1
        e, performance, pipe = execute_and_evaluate_code_block(code)

        if isinstance(performance, float):
            valid_pipeline = True
            pipeline_sentence = f"The code was executed and generated a dataset transformed with score {performance}"
        else:
            valid_pipeline = False
            pipeline_sentence = "The last code did not generate a valid preprocessing pipeline"

        display_method(
            "\n"
            + f"*Iteration {i}*\n"
            + f"*Valid pipeline: {str(valid_pipeline)}*\n"
            + f"```python\n{format_for_display(code)}\n```\n"
            + f"Performance {performance} \n"
            + f"{pipeline_sentence}\n"
            + f"\n"
        )

        if e is not None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Write the data to a CSV file
            with open(f'pipelines_{identifier}.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, code, e])
            general_considerations = possible_errors(str(e))
            if task == 'classification':
                additional_data = ''
            else:
                additional_data = ''

            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Code execution failed, error type: {type(e)}, error: {str(e)}.\n 
                    Code: ```python{code}```\n. 
                    {general_considerations} \n
                    {additional_data} \n
                    Generate the pipeline fixing the error, Let’s think step by step. \n:
                                ```python
                                """,
                },
            ]
            continue

        if e is None:
            list_codeblocks.append(code) # We are going to run this code if it is working
            list_performance.append(performance)
            print('The performance of this pipeline is: ', performance)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Write the data to a CSV file
            with open(f'pipelines_{identifier}.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, code, str(performance)])
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""The preprocessing pipeline "{code}" provided a score of "{performance}".  
                    Again, here are the similar Pipelines:
                    "
                    {similar_pipelines}
                    "
                    Generate the next Pipeline, it should not be identical to previous iterations. 
                    Yet, you could take inspiration from the pipelines you have previously generated to improve them further (hyperparameter optimization). 
        Next codeblock:
        """,
                },
            ]

    return code, prompt, messages, list_codeblocks, list_performance

