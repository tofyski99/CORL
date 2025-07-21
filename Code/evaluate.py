# -*- coding: utf-8 -*-
# @Author  : n13eho
# @Time    : 2024.03.25

"""
Evaluate models on evaluation datasets in detail.
"""

import glob
import json
import os
import numpy as np
from tqdm import tqdm
import onnxruntime as ort

current_dir = os.path.split(os.path.abspath(__file__))[0]
project_root_path = current_dir.rsplit('/', 1)[0]

if __name__ == "__main__":

    data_dir = os.path.join(project_root_path, 'ALLdatasets', 'evaluate', 'v2')  # < modify the path to your data
    onnx_models = ['ourmodel', 'baseline', 'Schaferct_model']  # Only include the models you want to compare
    onnx_models_dir = os.path.join(project_root_path, 'onnx_model_for_evaluation')
    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)

    # Define labels for each model in the plots
    model_labels = ['Our model', 'baseline', 'Schaferct Model']

    # Load ONNX models and get input names
    ort_sessions = []
    input_names_list = []
    for m in onnx_models:
        m_path = os.path.join(onnx_models_dir, m + '.onnx')
        ort_session = ort.InferenceSession(m_path)
        ort_sessions.append(ort_session)
        # Retrieve input names
        inputs = ort_session.get_inputs()
        input_names = [input.name for input in inputs]
        input_names_list.append(input_names)
        print(f"Model {m} input names: {input_names}")

    # Initialize latent factors if any model expects it
    latent_dim = 64  # Adjust according to your model
    latent_factors = np.zeros((1, latent_dim), dtype=np.float32)

    # Initialize lists to store errors
    mae_errors = {m: [] for m in onnx_models}
    mse_errors = {m: [] for m in onnx_models}

    # Initialize new metric trackers
    error_rates = {m: [] for m in onnx_models}
    over_estimation_rates = {m: [] for m in onnx_models}
    under_estimation_rates = {m: [] for m in onnx_models}

    # Begin evaluation
    for filename in tqdm(data_files, desc="Processing"):
        with open(filename, "r") as file:
            call_data = json.load(file)

        observations = np.asarray(call_data['observations'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)

        # Initialize predictions and states for all models
        model_predictions = {m: [] for m in onnx_models}
        hidden_states = {m: np.zeros((1, 1), dtype=np.float32) for m in onnx_models}
        cell_states = {m: np.zeros((1, 1), dtype=np.float32) for m in onnx_models}

        for t in range(observations.shape[0]):
            obss = observations[t:t+1, :].reshape(1, 1, -1)
            for idx, orts in enumerate(ort_sessions):
                m = onnx_models[idx]
                input_names = input_names_list[idx]
                feed_dict = {}
                if 'obs' in input_names:
                    feed_dict['obs'] = obss.astype(np.float32)
                if 'latent_factors' in input_names:
                    feed_dict['latent_factors'] = latent_factors
                if 'hidden_states' in input_names:
                    feed_dict['hidden_states'] = hidden_states[m]
                if 'cell_states' in input_names:
                    feed_dict['cell_states'] = cell_states[m]
                
                # Run the model
                outputs = orts.run(None, feed_dict)
                bw_prediction = outputs[0]
                # Update hidden and cell states if they are in the outputs
                if 'hidden_states' in input_names:
                    hidden_states[m] = outputs[1]
                if 'cell_states' in input_names:
                    cell_states[m] = outputs[2]
                model_predictions[m].append(bw_prediction[0, 0, 0])

        # Convert predictions to numpy arrays
        for m in onnx_models:
            model_predictions[m] = np.asarray(model_predictions[m], dtype=np.float32)

            # Create a combined valid indices mask where both arrays are not NaN
            valid_indices = (~np.isnan(true_capacity)) & (~np.isnan(model_predictions[m]))

            # Filter both arrays using the combined valid_indices
            true_capacity_filtered = true_capacity[valid_indices]
            model_predictions_filtered = model_predictions[m][valid_indices]

            if len(true_capacity_filtered) == 0:
                print(f"No valid data to evaluate for model {m} in file: {filename}")
                continue

            # Calculate all metrics
            mae = np.mean(np.abs(model_predictions_filtered - true_capacity_filtered))
            mse = np.mean((model_predictions_filtered - true_capacity_filtered) ** 2)
            
            # Calculate error rate (1 - accuracy)
            accuracies = np.maximum(0, 1 - np.abs(model_predictions_filtered - true_capacity_filtered) / true_capacity_filtered)
            error_rate = 1 - np.mean(accuracies)
            
            # Calculate over/under estimation rates
            over_estimates = np.sum(model_predictions_filtered > true_capacity_filtered) / len(true_capacity_filtered)
            under_estimates = np.sum(model_predictions_filtered < true_capacity_filtered) / len(true_capacity_filtered)
            
            # Store all metrics
            mae_errors[m].append(mae)
            mse_errors[m].append(mse)
            error_rates[m].append(error_rate)
            over_estimation_rates[m].append(over_estimates)
            under_estimation_rates[m].append(under_estimates)

    # Calculate averages across all evaluation cases
    avg_mae = {m: np.mean(mae_errors[m]) for m in onnx_models}
    avg_mse = {m: np.mean(mse_errors[m]) for m in onnx_models}
    avg_error_rate = {m: np.mean(error_rates[m]) for m in onnx_models}
    avg_over_rate = {m: np.mean(over_estimation_rates[m]) for m in onnx_models}
    avg_under_rate = {m: np.mean(under_estimation_rates[m]) for m in onnx_models}

    # Print detailed results with units
    for m in onnx_models:
        print(f"\nModel: {m}")
        print(f"Average MAE: {avg_mae[m]:.2f} Mbps")
        print(f"Average MSE: {avg_mse[m]:.2f} Mbps²")
        print(f"Error Rate: {avg_error_rate[m]:.2%}")
        print(f"Over-estimation Rate: {avg_over_rate[m]:.2%}")
        print(f"Under-estimation Rate: {avg_under_rate[m]:.2%}")
