from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from evaluate import load
from datasets import load_dataset
import wandb
import numpy as np
import os

@dataclass
class CustomArguments:
    """
    Custom arguments for the training script.
    """
    max_train_samples: int = field(
        default=-1, metadata={"help": "Number of samples to use for training"}
    )
    max_eval_samples: int = field(
        default=-1, metadata={"help": "Number of samples to use for evaluation"}
    )
    max_predict_samples: int = field(
        default=-1, metadata={"help": "Number of samples to use for prediction"}
    )
    model_path: str = field(
        default=None, metadata={"help": "Path to pretrained model for prediction"}
    )
    batch_size: int = field(
        default=64
        , metadata={"help": "Batch size for training"}
    )
    lr: float = field(
        default=1e-5, metadata={"help": "Learning rate for training"}
    )
    experiment_name: str = field(
        default="my_experiment", metadata={"help": "Name of the experiment"}
    )

def set_arguments(training_args: TrainingArguments, custom_args: CustomArguments):
    """
    Set the arguments for the training script.
    Args:
        training_args: The training arguments for the model.
        custom_args: The custom arguments for the training script.

    Returns:

    """
    training_args.per_device_train_batch_size = custom_args.batch_size
    training_args.per_device_eval_batch_size = 64
    training_args.learning_rate = custom_args.lr
    training_args.run_name = custom_args.experiment_name
    if not os.path.exists("logs"):
        os.makedirs("logs")

    training_args.logging_dir = "logs/" + custom_args.experiment_name
    training_args.logging_strategy = 'steps'
    training_args.logging_steps = 1
    training_args.report_to = "wandb"
    training_args.save_strategy = "no"

def init_wandb(training_args: TrainingArguments, custom_args: CustomArguments):
    """
    Initialize wandb for logging the experiment.
    Args:
        training_args: The training arguments for the model.
        custom_args: The custom arguments for the training script.

    Returns:

    """
    wandb.login()
    wandb.init(
        project="ANLP-Ex1",
        entity = "odvir-hebrew-university-of-jerusalem",
        name=custom_args.experiment_name,
        config={
            "batch_size": custom_args.batch_size,
            "learning_rate": custom_args.lr,
            "epochs": training_args.num_train_epochs,
            "max_train_samples": custom_args.max_train_samples,
            "max_eval_samples": custom_args.max_eval_samples,
            "max_predict_samples": custom_args.max_predict_samples,

        },
    )


def create_dataset(custom_args: CustomArguments):
    """
    Create the dataset for the experiment.
    :param custom_args: the custom arguments for the training script
    :return: the dataset
    """
    # load the dataset
    raw_dataset = load_dataset("glue", "mrpc")

    # Limit the dataset to the specified number of samples
    raw_dataset["train"] = raw_dataset["train"] if custom_args.max_train_samples == -1 else raw_dataset["train"].select(range(custom_args.max_train_samples))
    raw_dataset["validation"] = raw_dataset["validation"] if custom_args.max_eval_samples == -1 else raw_dataset["validation"].select(range(custom_args.max_eval_samples))
    raw_dataset["test"] = raw_dataset["test"] if custom_args.max_predict_samples == -1 else raw_dataset["test"].select(range(custom_args.max_predict_samples))

    return raw_dataset


def tokenize_dataset(dataset, tokenizer, prediction=False):
    """
    Tokenize the dataset using the provided tokenizer.
    :param dataset: the dataset to tokenize
    :param tokenizer: the tokenizer to use for tokenization
    :param prediction: whether to use the prediction tokenizer
    :return: the tokenized dataset
    """

    # Define the preprocessing function
    def preprocess_function(examples):
        """
        Preprocess the input examples for the model.
        Args:
            examples: The input examples to preprocess.
        Returns:
            The preprocessed examples.
        """
        # Tokenize the input text
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding=True if not prediction else False,
            max_length=512,
        )

    # preprocess the dataset
    return dataset.map(preprocess_function, batched=True)



def create_trainer(training_args: TrainingArguments, model, tokenized_dataset, tokenizer):
    """
    Create a Trainer instance for training and evaluating the model.
    Args:
        training_args: The training arguments for the model.
        model: The model to train and evaluate.
        tokenized_dataset : The tokenized dataset for training and evaluation.
        tokenizer: The tokenizer to use for the model.

    Returns:
        The Trainer instance.
    """
    # Load evaluation metric
    metric = load("accuracy")

    def compute_metrics(eval_predictions):
        """
        Compute the evaluation metrics for the model.
        Args:
            eval_predictions: The predictions and labels from the evaluation.

        Returns:
            The computed metrics.
        """
        logits, labels = eval_predictions
        predictions = np.argmax(logits, axis=1)
        result = metric.compute(predictions=predictions, references=labels)
        return result

    # Set training arguments
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )


def train_and_evaluate(training_args: TrainingArguments, custom_args: CustomArguments):
    """
    Train and evaluate the model using the provided training arguments and custom arguments.
    Args:
        training_args: The training arguments for the model.
        custom_args: The custom arguments for the training script.

    Returns:

    """
    # login to wandb and initialize the run
    init_wandb(training_args, custom_args)

    # Load the dataset
    raw_dataset = create_dataset(custom_args)

    # Load model and tokenizer
    config = AutoConfig.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        config=config,
    )

    # tokenize the dataset
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)

    # create the trainer of the model
    trainer = create_trainer(training_args, model, tokenized_dataset, tokenizer)

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the model (create a directory if it doesn't exist)
    if not os.path.exists("models"):
        os.makedirs("models")

    model.save_pretrained("models/" + custom_args.experiment_name)

    # Log the model to wandb
    wandb.finish()


def predict(training_args: TrainingArguments, custom_args: CustomArguments):
    """
    Predict the test dataset using the trained model.
    :param training_args: The training arguments for the model.
    :param custom_args: The custom arguments for the training script.
    :return:
    """
    # Load the model and tokenizer for prediction
    config = AutoConfig.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(custom_args.model_path, config=config)
    model.eval()

    # Load the test dataset
    raw_dataset = create_dataset(custom_args)

    # Tokenize the test dataset
    tokenized_test_dataset = tokenize_dataset(raw_dataset, tokenizer, prediction=True)

    # Create a trainer for prediction
    trainer = create_trainer(training_args, model, tokenized_test_dataset, tokenizer)

    # predict the test dataset
    full_predictions = trainer.predict(tokenized_test_dataset["test"])
    predictions = np.argmax(full_predictions.predictions, axis=1)

    # Save the predictions to a file
    with open("predictions.txt", "w") as f:
        for pred, example in zip(predictions, raw_dataset["test"]):
            # add sentences
            f.write(f"{example['sentence1']}###{example['sentence2']}###{pred}\n")

    # print the test accuracy
    print({"Test Accuracy": round(full_predictions.metrics["test_accuracy"], 4)})



def main():
    """
    Main function of the experiment script.
    """

    # Load custom arguments and training arguments
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    custom_args, training_args = parser.parse_args_into_dataclasses()
    # Set the training arguments
    set_arguments(training_args, custom_args)

    if training_args.do_train:
        train_and_evaluate(training_args, custom_args)

    if training_args.do_predict:
        predict(training_args, custom_args)




if __name__ == "__main__":
    main()
