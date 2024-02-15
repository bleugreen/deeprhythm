import pandas as pd
import os
import librosa
import essentia.standard as es
import time
import sys
sys.path.append('/home/bleu/ai/deeprhythm/src')

from deeprhythm.model.infer import predict_global_bpm, make_kernels, load_cnn_model, predict_global_bpm_cont



def estimate_tempo_essentia_multi(audio_path):
    audio = es.MonoLoader(filename=audio_path)()
    extractor_multi = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = extractor_multi(audio)
    print(bpm)
    return bpm

def estimate_tempo_essentia_percival(audio_path):
    audio = es.MonoLoader(filename=audio_path)()
    bpm = es.PercivalBpmEstimator()(audio)
    print(bpm)
    return bpm

def estimate_tempo_essentia_degara(audio_path):
    audio = es.MonoLoader(filename=audio_path)()
    extractor_deg = es.RhythmExtractor2013(method="degara")
    bpm, beats, beats_confidence, _, beats_intervals = extractor_deg(audio)
    print(bpm)
    return bpm

def estimate_tempo_librosa(audio_path):
    audio, _ = librosa.load(audio_path, sr=22050)
    bpm = librosa.beat.tempo(y=audio, sr=22050)[0]
    print(bpm)
    return bpm

def estimate_tempo_cnn(audio_path, model, specs):
    bpm= predict_global_bpm(audio_path, model=model, specs=specs)[0]
    print(bpm)
    return bpm

def estimate_tempo_cnn_cont(audio_path, model, specs):
    bpm= predict_global_bpm_cont(audio_path, model=model, specs=specs)[0]
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
            if row['source'] == 'fma':
                continue
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
    print('Test Songs:', len(results['DeepRhythm (cpu)']['times']))
    for method, metrics in results.items():
        accuracy1 = sum(metrics['accuracy1']) / len(metrics['accuracy1']) * 100
        accuracy2 = sum(metrics['accuracy2']) / len(metrics['accuracy2']) * 100

        avg_time = sum(metrics['times']) / len(metrics['times'])
        print('-----'*20)
        print(f"{method:<18}: Acc1 = {accuracy1:.2f}%, Acc2 = {accuracy2:.2f}%, Avg Time = {avg_time:.4f}s, Total={sum(metrics['times']):.2f}s")

if __name__ == '__main__':
    test_set = pd.read_csv('/media/bleu/bulkdata2/deeprhythmdata/test.csv')

    cpu_model = load_cnn_model(device='cpu')
    cpu_specs = make_kernels(device='cpu')

    cuda_model = load_cnn_model(device='cuda')
    cuda_specs = make_kernels(device='cuda')

    # Define the estimation methods
    methods = {
        'Essentia (multi)': lambda audio_path: estimate_tempo_essentia_multi(audio_path),
        'Essentia (percival)':estimate_tempo_essentia_percival,
        'Essentia (degara)': lambda audio_path: estimate_tempo_essentia_degara(audio_path),
        'Librosa': estimate_tempo_librosa,
        'DeepRhythm (cuda)': lambda audio_path: estimate_tempo_cnn(audio_path, cuda_model, cuda_specs),
        'DeepRhythm (cpu)': lambda audio_path: estimate_tempo_cnn(audio_path, cpu_model, cpu_specs),


    }

    # Run the benchmark
    results = run_benchmark(test_set, methods)

    # Generate the report
    generate_report(results)
