import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
def generate_data(N, M, input_type, output_type):    
    # Generate input features X
    if input_type == "real":
        features = pd.DataFrame(np.random.randn(N, M))  # real-valued features
    elif input_type == "discrete":
        features = pd.DataFrame(
            {i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)}
        )  # categorical features

    # Generate output labels y
    if output_type == "real":
        labels = pd.Series(np.random.randn(N))  # regression target
    elif output_type == "discrete":
        labels = pd.Series(np.random.randint(M, size=N), dtype="category")  # classification target

    return features, labels



# Function to calculate average time (and std) taken by fit() and predict() 
# for different N and M for 4 different cases of DTs
def evaluate_runtime(N, M, input_type, output_type, test_size, criterias, num_average_time):
    X, y = generate_data(N, M, input_type, output_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    time_results = {}  # stores timing results for each criteria
    
    for criterion in criterias:
        train_durations = []   # stores training times
        test_durations = []    # stores prediction times
        
        for _ in range(num_average_time):
            tree_model = DecisionTree(criterion=criterion, max_depth=5)

            # Measure training time
            start_train = time.time()
            tree_model.fit(X_train, y_train)
            train_durations.append(time.time() - start_train)

            # Measure prediction time
            start_test = time.time()
            y_pred = tree_model.predict(X_test)
            test_durations.append(time.time() - start_test)

        # Compute averages and standard deviations
        avg_train_duration = np.mean(train_durations)
        avg_test_duration = np.mean(test_durations)
        std_train_duration = np.std(train_durations)
        std_test_duration = np.std(test_durations)

        # Print summary
        print(f"    Criteria: {criterion}")
        print(f"        Average Training Time: {avg_train_duration:.4f} seconds (std: {std_train_duration:.4f})")
        print(f"        Average Prediction Time: {avg_test_duration:.4f} seconds (std: {std_test_duration:.4f})")

        # Store in dictionary
        time_results[criterion] = {
            "train_time": avg_train_duration,
            "test_time": avg_test_duration,
            "std_train_time": std_train_duration,
            "std_test_time": std_test_duration
        }

    return time_results


def run_n_m(N_values, M_values, input_types, output_types, test_size, criterias, num_average_time):
    results = {}
    for N in N_values:
        for M in M_values:
            print(f"\nEvaluating for N={N}, M={M}\n")
            results[(N, M)] = {}
            for input_type in input_types:
                for output_type in output_types:
                    print(f"    Input Type: {input_type}, Output Type: {output_type}")
                    single_case_result = evaluate_runtime(
                        N, M, input_type, output_type, test_size, criterias, num_average_time
                    )
                    results[(N, M)][(input_type, output_type)] = single_case_result
                    print()
            print("=" * 50)
    return results


# Function to plot the results
def plot_time_complexity_separate(results, N_values, M_values, criteria):
    # ------------------------- Time vs N -------------------------
    print("Time vs Number of Samples (N)")
    plt.figure(figsize=(14, 5*len(M_values)))
    plt.suptitle(f"Time vs Number of Samples (N) - Criteria: {criteria}", y=1, fontsize=16)

    for idx, M in enumerate(M_values):
        ax_train = plt.subplot(len(M_values), 2, 2*idx + 1)
        ax_test = plt.subplot(len(M_values), 2, 2*idx + 2)

        for input_type in ["discrete", "real"]:
            for output_type in ["discrete", "real"]:
                train_times = []
                test_times = []

                for N in N_values:
                    case_data = results[(N, M)][(input_type, output_type)][criteria]
                    train_times.append(case_data["train_time"])
                    test_times.append(case_data["test_time"])
                
                ax_train.plot(N_values, train_times, marker='o', label=f'{input_type.capitalize()}-{output_type.capitalize()}')
                ax_test.plot(N_values, test_times, marker='o', label=f'{input_type.capitalize()}-{output_type.capitalize()}')

        ax_train.set_xlabel("Number of Samples (N)")
        ax_train.set_ylabel("Training Time (seconds)")
        ax_train.set_title(f"Training Time vs Number of Samples (N), M = {M}")
        ax_train.legend()
        ax_train.grid(True)

        ax_test.set_xlabel("Number of Samples (N)")
        ax_test.set_ylabel("Prediction Time (seconds)")
        ax_test.set_title(f"Prediction Time vs Number of Samples (N), M = {M}")
        ax_test.legend()
        ax_test.grid(True)

    plt.savefig(f"./Decision Tree Implementation/1.4 Data/1.4_time_vs_N_{criteria}.png", bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    plt.show()

    print("=" * 50)

    # ------------------------- Time vs M -------------------------
    print("Time vs Number of Features (M)")
    plt.figure(figsize=(14, 5*len(N_values)))
    plt.suptitle(f"Time vs Number of Features (M) - Criteria: {criteria}", y=1, fontsize=16)

    for idx, N in enumerate(N_values):
        ax_train = plt.subplot(len(N_values), 2, 2*idx + 1)
        ax_test = plt.subplot(len(N_values), 2, 2*idx + 2)

        for input_type in ["discrete", "real"]:
            for output_type in ["discrete", "real"]:
                train_times = []
                test_times = []
                
                for M in M_values:
                    case_data = results[(N, M)][(input_type, output_type)][criteria]
                    train_times.append(case_data["train_time"])
                    test_times.append(case_data["test_time"])

                ax_train.plot(M_values, train_times, marker='o', label=f'{input_type.capitalize()}-{output_type.capitalize()}')
                ax_test.plot(M_values, test_times, marker='o', label=f'{input_type.capitalize()}-{output_type.capitalize()}')

        ax_train.set_xlabel("Number of Features (M)")
        ax_train.set_ylabel("Training Time (seconds)")
        ax_train.set_title(f"Training Time vs Number of Features (M), N = {N}")
        ax_train.legend()
        ax_train.grid(True)

        ax_test.set_xlabel("Number of Features (M)")
        ax_test.set_ylabel("Prediction Time (seconds)")
        ax_test.set_title(f"Prediction Time vs Number of Features (M), N = {N}")
        ax_test.legend()
        ax_test.grid(True)

    plt.savefig(f"./ Decision Tree Implementation/1.4 Data/1.4_time_vs_M_{criteria}.png", bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    plt.show()


# ---------------------- Run the experiments ----------------------
N_values = [50, 100, 500, 1000, 5000]
M_values = [1, 5, 10, 20, 50, 100]
criterias = ["information_gain", "gini_index"]
input_types = ["real", "discrete"]
output_types = ["real", "discrete"]
test_size = 0.3

results = run_n_m(N_values, M_values, input_types, output_types, test_size, criterias, num_average_time)


# Save the results to a file
import pickle
if not os.path.exists(r'./ Decision Tree Implementation/1.4 Data'):
    os.makedirs(r'./ Decision Tree Implementation/1.4 Data')
with open(r'./ Decision Tree Implementation/1.4 Data/1.4_results.pkl', 'wb') as f:
    pickle.dump(results, f)


# Plot the results
plot_time_complexity_separate(results, N_values, M_values, criteria="information_gain")
plot_time_complexity_separate(results, N_values, M_values, criteria="gini_index")
