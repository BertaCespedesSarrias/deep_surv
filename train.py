import configargparse
from torch.utils.tensorboard import SummaryWriter
from loader import load_data, get_dataloaders
from model_setup import choose_model
import pandas as pd
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import json
import joblib
import torch
from loss import custom_loss, custom_loss_time

def str2bool(x: str) -> bool:
    """Cast string to boolean.

    Useful for parsing command line arguments.
    """
    if not isinstance(x, str):
        raise TypeError("String expected.")
    elif x.lower() in ("true", "t", "1"):
        return True
    elif x.lower() in ("false", "f", "0"):
        return False
    else:
        raise ValueError(f"'{x}' does not seem to be boolean.")

def get_argparser():
    p = configargparse.ArgParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        allow_abbrev=False,
    )

    p.add(
        "-c",
        "--config",
        is_config_file=True,
        help="Config file path (other given arguments will superseed this).",
    )
    p.add("--name", type=str, default=None, help="Name of the training run.")
    p.add(
        "--model_type",
        type=str,
        default="cox",
        choices=["cox", "random_forest", "deep"],
    )
    p.add("-e", "--epochs", type=int, default=20)
    p.add("--batch_size", type=int, default=128)
    p.add("--seed", type=int, default=42)
    p.add(
        "--data_path",
        type=str,
        required=True,
        help="Path to csv file with patient data.",
    )
    p.add(
        "--map_path",
        type=str,
        required=True,
        help="Path to map between diseases and their code.",
    )
    p.add(
        "--train_split",
        type=float,
        default=0.85,
        help="Split of train and validation sets.",
    )
    p.add(
        "--test_split",
        type=float,
        default=0.85,
        help="Split of training (including train and validation sets) and test data.",
    )
    p.add("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    p.add("--learning_rate","-lr", type=float, default=1e-4)
    p.add(
        "--drop_last",
        type=str2bool,
        default=False,
        help="Flag to drop last batch if size is not equal to all other batches.",
    )
    p.add("--input_size", type=int, help='Input size for linear layer.')
    p.add("--hidden_size", type=int, help="Hidden size for linear layer.")
    p.add("--output_size", type=int, help='Output size for linear layer.')
    p.add(
        "-o",
        "--out_dir",
        type=str,
        default="metrics",
        help="Save metrics csv in this dir.",
    )
    p.add(
        "--log",
        type=str2bool,
        default=True,
        help="Flag for logging in tensorboard (useful when debugging - set to false.)",
    )
    p.add(
        "--gpu",
        "-g",
        type=str,
        default="1",
        help="GPUs to use. Can be a single integer, a comma-separated list of integers, or an interval `a-b`, or 'cpu'.",
    )
    p.add("-r", "--run", type=int, default=0, help="Run number for saving purposes.")
    p.add( "--num_diseases", "-n_disease", type=int, default = None, help='Number of diseases to run models on.')
    p.add("--features", "-feat", type=str, nargs="+",help="Name of features to be included in the survival analysis.")
    p.add("--disease_list", "-diseases", type=str, default = None, nargs="+",help="List of disease names to be included in the survival analysis. If provided, num_diseases shouldn't.")
    p.add("--fill_nan_cols", type=str, nargs='+', help = "Name of feature column where NaN values should be substituted by mean/median.")
    p.add("--hyper_tuning_rsf", type = str2bool, default=False, help="Flag to apply hyperparameter tuning on Random Survival Forest and return best parameters. Otherwise run with default.")
    p.add("--n_estimators", type = int, nargs='+', help = "Number of trees in forest. Provide for Random Survival Forest, otherwise default.")
    p.add("--max_depth", type = int, nargs='+', help = "Maximum depth of each tree. Provide for Random Survival Forest, otherwise default.")
    p.add("--max_features", type = int, nargs='+', help = "Number of features considered when looking for best split. Provide for Random Survival Forest, otherwise default.")
    p.add("--min_samples_split", type = int, nargs='+', help = "Minimum number of samples to split an internal node. Provide for Random Survival Forest, otherwise default.")
    p.add("--min_samples_leaf", type = int, nargs='+', help = "Minimum number of samples required to be at a leaf node. Provide for Random Survival Forest, otherwise default.")
    p.add("--time_max", "-t_max", type=int, help = "Maximum time to event present ")
    p.add("--norm_mode", type=str, default='standard', choices=['standard', 'normal'], help= "Data normalization mode: 'standard': Zero mean unit variance or 'normal': Min-max normalization")
    p.add("--cv_folds", type=int, default=5, help="Folds for cross validation.")
    return p

# def get_param_grid(args):
#     param_grid = {}
#     param_grid['n_estimators'] = [2]
#     param_grid['max_depth'] = [2, 3]
#     param_grid['max_features'] = ['sqrt']
#     param_grid['min_samples_leaf'] = [1]
#     return param_grid

def get_param_grid(args):
    param_grid = {}
    if args.n_estimators:
        param_grid['n_estimators'] = args.n_estimators
    else:
        param_grid['n_estimators'] = [50, 100, 200, 300]
    if args.max_depth:
        param_grid['max_depth'] = args.max_depth
    else:
        param_grid['max_depth'] = [None, 5, 10, 20, 30]
    if args.max_features:
        param_grid['max_features'] = args.max_features
    else:
        param_grid['max_features'] = ['sqrt', 'log2']
    if args.min_samples_split:
        param_grid['min_samples_split'] = args.min_samples_split
    else:
        param_grid['min_samples_split'] = [2, 5, 10]
    if args.min_samples_leaf:
        param_grid['min_samples_leaf'] = args.min_samples_leaf
    else:
        param_grid['min_samples_leaf'] = [1, 2, 4]
    return param_grid

def concordance_index_scorer(y_true, y_pred):
    event, time = y_true['event'], y_true['time']
    score = concordance_index_censored(event, time, y_pred)
    return score[0]

def main(args):
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == 'deep':
        train_data, val_data, test_data, disease_ids, num_feat, time_span = load_data(args)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args, train_data, val_data, test_data,)
        
        num_diseases = len(disease_ids)
        model = choose_model(args, device, num_feat, num_diseases, time_span)
        
        metrics = model.train(train_dataloader, val_dataloader, custom_loss)
        # metrics, loss = trainer.train(args.epochs, train_dataloader, val_dataloader, out_dir = args.out_dir, learning_rate = args.learning_rate, log = args.log)
    
    elif args.model_type == 'deep_time':
        train_data, val_data, test_data, disease_ids, num_feat, time_span = load_data(args)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args, train_data, val_data, test_data,)
        
        num_diseases = len(disease_ids)
        model = choose_model(args, device, num_feat, num_diseases)
        
        metrics = model.train(train_dataloader, val_dataloader, custom_loss_time)
        
    else:
        model = choose_model(args)
        train_data, test_data, disease_ids = load_data(args)
        evaluation = pd.DataFrame(index = disease_ids, columns=['concordance_index', 'concordant_pairs', 'discordant_pairs', 'tied_pred','tied_time'])
        # Concordance index: Model predictive accuracy based on the pairwise comparison of subjects' predicted survivial times versus their actual outcomes.
        # It models the model's ability to correctly rank pairs of subjects in terms of their survival times.
        # Concordant/Diconcordant pairs: Those pairs of subjects for which the predicted and actual outcomes are/aren't in agreements regarding their order.
        # Tied pairs: Survival analysis predicts same survival time for two or more subjects, but in reality they experience the event at different times.
        # Tied events: Situations where two or more subjects experience the event of interest at exactly the same time, but the model predicts different survival times for them.
        
        # Do we need three way split?
        (tr_data, tr_label) = train_data
        (te_data, te_label) = test_data
        
        for tr_label_disease, te_label_disease, id in zip(tr_label, te_label, disease_ids):
            if args.hyper_tuning_rsf == True:
                print('hey')
                print(model)
                assert args.model_type == 'random_forest', "Please set model_type to random_forest in config file for RSF hyperparameter tuning."
                param_grid = get_param_grid(args)
                grid_search = GridSearchCV(estimator=model.model, param_grid = param_grid, cv=args.cv_folds, verbose=2, scoring=make_scorer(concordance_index_scorer, greater_is_better=True))
                grid_search.fit(tr_data, tr_label_disease)
                with open(f"best_params_rsf_{id}.jsonl", "w") as file:
                    data = {
                        "Best parameters": grid_search.best_params_,
                        "Best score": grid_search.best_score_
                    }
                    file.write(json.dumps(data) + "\n")
                best_model = grid_search.best_estimator_
                joblib.dump(best_model, f'rsf_{id}.joblib')

            else:
                model.train(tr_data, tr_label_disease)
                pred = model.predict(te_data)
                score = concordance_index_censored(te_label_disease[f'event'], te_label_disease[f'time'], pred)
                
                evaluation.loc[id] = score
                import pdb
                pdb.set_trace()

                evaluation.to_csv(f'concordance_index_{args.model_type}.csv', index=True)

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)