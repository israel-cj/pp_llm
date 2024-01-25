from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code
from .pp_llm import generate_dataset, list_pipelines
from .run_llm_code import run_llm_code
from .llmoptimization import optimize_LLM
from .llmensemble import generate_code_embedding, generate_ensemble_manually
from typing import Union
import numpy as np
import pandas as pd
import stopit
import uuid

class PP_LLM():
    """
    Parameters:
    """
    def __init__(
            self,
            iterations: int = 10,
            llm_model: str = "gpt-3.5-turbo",
            task="classification",
            max_total_time = 180,
    ) -> None:
        self.llm_model = llm_model
        self.iterations = iterations
        self.task = task
        self.timeout = max_total_time
        self.uid = str(uuid.uuid4())

    def fit(
            self, X, y,
    ):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.

        """
        # Generate a unique UUID
        print('uid', self.uid)

        if self.task == "classification":
            y_ = y.squeeze() if isinstance(y, pd.DataFrame) else y
            self._label_encoder = LabelEncoder().fit(y_)
            if any(isinstance(yi, str) for yi in y_):
                # If target values are `str` we encode them or scikit-learn will complain.
                y = self._label_encoder.transform(y_)
                self._decoding = True

        self.X = X
        self.y = y
        try:
            with stopit.ThreadingTimeout(self.timeout):
                self.code, prompt, messages, list_codeblocks_generated, list_performance_pipelines = generate_dataset(
                    self.X,
                    self.y,
                    model=self.llm_model,
                    iterative=self.iterations,
                    display_method="markdown",
                    task = self.task,
                    identifier=self.uid,
                )
            if len(list_pipelines)>0:
                index_best_pipeline = list_performance_pipelines.index(max(list_performance_pipelines))
                best_code = list_pipelines[index_best_pipeline] # We get at least 1 pipeline to return

            get_pipelines = list_pipelines
            if len(get_pipelines) == 0:
                raise ValueError("Not pipeline could be created")

            transformed_df = run_llm_code(
                best_code,
                self.X,
                self.y,
            )

            return transformed_df, self.y

        except stopit.TimeoutException:
            print("Timeout expired")





