import pandas as pd
import os
from tempocnn.classifier import TempoClassifier
from tempocnn.feature import read_features
import time

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
def estimate_tempo_cnn(audio_path, model):
    features = read_features(audio_path)
    bpm = model.estimate_tempo(features, interpolate=False)
    print(bpm)
    return bpm


def is_within_tolerance(predicted_bpm, true_bpm, tolerance=0.02, multiples=[1]):
    for multiple in multiples:
        if true_bpm * multiple * (1 - tolerance) <= predicted_bpm <= true_bpm * multiple * (1 + tolerance):
            return True
    return False


def run_benchmark(test_set, estimation_methods):
    results = {method: {'times': [], 'accuracy1': [], 'accuracy2':[]} for method in estimation_methods}
    for method_name, method_func in estimation_methods.items():
        for _, row in test_set.iterrows():
            true_bpm = row['bpm']
            audio_path = os.path.join('/media/bleu/bulkdata2/deeprhythmdata', row['filename'])
            start_time = time.time()
            predicted_bpm = method_func(audio_path)
            elapsed_time = time.time() - start_time
            results[method_name]['times'].append(elapsed_time)
            correct1 = is_within_tolerance(predicted_bpm, true_bpm)
            results[method_name]['accuracy1'].append(correct1)
            correct2 = is_within_tolerance(predicted_bpm, true_bpm, multiples=[0.5, 1, 2, 3])
            results[method_name]['accuracy2'].append(correct2)

    return results


def generate_report(results):
    for method, metrics in results.items():
        accuracy1 = sum(metrics['accuracy1']) / len(metrics['accuracy1']) * 100
        accuracy2 = sum(metrics['accuracy2']) / len(metrics['accuracy2']) * 100

        avg_time = sum(metrics['times']) / len(metrics['times'])
        print('-----'*20)
        print(f"{method:<18}: Acc1 = {accuracy1:.2f}%, Acc2 = {accuracy2:.2f}%, Avg Time = {avg_time:.4f}s, Total={sum(metrics['times']):.2f}s")

if __name__ == '__main__':
    test_set = pd.read_csv('/media/bleu/bulkdata2/deeprhythmdata/test.csv')
    fcn_model = TempoClassifier('fcn')
    cnn_model = TempoClassifier('cnn')

    # Define the estimation methods
    methods = {
        'TempoCNN (cnn)': lambda audio_path: estimate_tempo_cnn(audio_path, cnn_model),
        'TempoCNN (fcn)': lambda audio_path: estimate_tempo_cnn(audio_path, fcn_model),
    }

    # Run the benchmark
    results = run_benchmark(test_set, methods)

    # Generate the report
    generate_report(results)
