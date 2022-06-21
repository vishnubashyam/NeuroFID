import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="Process Experiment Parameters")

    exp_parser = parser.add_argument_group("Experiment Parameters")
    exp_parser.add_argument("--experiment", type=str, help="Name of the experiment")
    exp_parser.add_argument(
        "--experiment_type",
        type=str,
        choices=["Classification", "Regression"],
        help="Type of experiment",
    )
    exp_parser.add_argument(
        "--leave_site_out",
        type=bool,
        default=False,
        help="Part of the experiment that uses leave-site-out validation",
    )
    exp_parser.add_argument(
        "--lso_main_task",
        type=str,
        help="Main task of the leave-site-out experiment",
    )
    exp_parser.add_argument(
        "--lso_task_type",
        type=str,
        choices=["Main", "Subtask", 'Baseline'],
        help="Main or subtask of the leave-site-out experiment",
    )
    exp_parser.add_argument(
        "--experiment_tag", type=str, default="None", help="Optional Tag for experiment"
    )
    exp_parser.add_argument(
        "--prediction_endpoint", type=str, help="Name of the prediction target"
    )
    exp_parser.add_argument(
        "--data_csv", type=str, help="Path to the csv file containing the data"
    )
    exp_parser.add_argument("--data_directory", type=str, help="Path to the NIFTI data")
    exp_parser.add_argument(
        "--dataloader_num_processes",
        type=int,
        help="Number of processes to use for dataloading",
    )
    exp_parser.add_argument(
        "--num_splits",
        type=int,
        default=4,
        help="Number of splits to use for cross validation",
    )
    exp_parser.add_argument(
        "--val_split_pct",
        type=float,
        default=0.3,
        help="Percentage of train split to use for validation",
    )

    model_parser = parser.add_argument_group("Model Parameters")
    model_parser.add_argument("--model_name", type=str, default="ResNet")
    model_parser.add_argument(
        "--model_size", type=int, help="Size of the model - 10, 18, 34, 50"
    )
    model_parser.add_argument(
        "--pretrained_weights", action="store_true", help="Use pretrained weights"
    )

    model_parser = parser.add_argument_group("Training Parameters")
    model_parser.add_argument("--batch_size", type=int, help="Size of the batch")
    model_parser.add_argument(
        "--max_epochs", type=int, help="Number of epochs to train for"
    )
    model_parser.add_argument(
        "--optimizer",
        type=str,
        choices=["Adam", "SGD", "RMSprop", "Adamax"],
        help="Optimizer to use: Adam, SGD, RMSprop, Adamax",
    )
    model_parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate before updating",
    )
    model_parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.25,
        help="How often to validate the model",
    )
    model_parser.add_argument(
        "--swa_epoch_start",
        type=float,
        default=0.6,
        help="Fraction of epochs before starting Stochastic Weight Averaging",
    )
    model_parser.add_argument(
        "--learning_rate", type=float, default=0.0006, help="Learning rate"
    )
    model_parser.add_argument(
        "--mixed_precision", action="store_true", help="Enable 16bit mixed precision"
    )

    return parser
