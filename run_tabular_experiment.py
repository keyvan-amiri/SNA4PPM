import argparse
import numpy as np
import yaml

from tabular_learning.modeling_helpers import get_config, get_results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run experiment with a given config file.")
    
    # Required positional arguments
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    
    # Optional arguments
    parser.add_argument("--exp_name", type=str, default="5CV-grouped", help="Experiment Name")
    parser.add_argument("--gpus", type=str, default="None", help="GPU IDs")
    parser.add_argument("--seeds_parallel", type=int, default=1, help="Number of parallel seeds")
    parser.add_argument("--val_strategy", type=str, default="5CV-grouped", help="Validation split type")
    parser.add_argument("--fe_type", type=str, default=None, help="Feature engineering type")
    parser.add_argument("--fe_order", type=int, default=0, help="Feature engineering order")
    parser.add_argument("--num_scaler", type=str, default="quantile", help="Numeric feature scaling method")
    parser.add_argument("--n_trials", type=int, default=0, help="Number of trials")
    parser.add_argument("--save_interval", type=int, default=5, help="Save interval for model checkpoints")

    args = parser.parse_args()

    args.fe_type = None if args.fe_type in ["", "None"] else args.fe_type
    args.fe_order = None if args.fe_order == 0 else int(args.fe_order)
    
    # Load configurations
    configs = get_config(args.model_name)
    configs["exp_name"] = args.val_strategy
    configs["dataset"]["val_strategy"] = args.val_strategy
    configs["dataset"]["dataset_name"] = args.dataset_name
    configs["dataset"]["fe_type"] = args.fe_type
    configs["dataset"]["fe_order"] = args.fe_order
    configs["dataset"]["num_scaler"] = args.num_scaler
    configs["model"]["gpus"] = args.gpus
    configs["model"]["seeds_parallel"] = args.seeds_parallel
    configs["hpo"]["n_trials"] = args.n_trials
    configs["hpo"]["save_interval"] = args.save_interval

    print(configs)
    
    # Run experiment
    res = get_results(configs)

    # Print performance metrics
    print(f"Performance:")
    test_perf = np.array(list(res["performance"]["Test"].values()))
    print("Performance", test_perf.mean().round(3), test_perf.std().round(3))
