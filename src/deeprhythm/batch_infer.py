import json
import os
import torch.multiprocessing as multiprocessing
import torch
import warnings
import time
import argparse
from deeprhythm.utils import load_and_split_audio
from deeprhythm.audio_proc.hcqm import make_kernels, compute_hcqm
from deeprhythm.model.infer import load_cnn_model
from deeprhythm.utils import class_to_bpm
from deeprhythm.utils import get_device


NUM_WORKERS = 4
NUM_BATCH = 256


def producer(task_queue, result_queue, completion_event, queue_condition, queue_threshold=NUM_BATCH*2):
    """
    Loads audio, splits it into a list of 8s clips, and puts the clips into the result queue.
    """
    while True:
        task = task_queue.get()
        if task is None:
            result_queue.put(None)  # Send termination signal to indicate this producer is done
            completion_event.wait()  # Wait for the signal to exit
            break
        filename = task
        with queue_condition:  # Use the condition to wait if the queue is too full before loading audio
            while result_queue.qsize() >= queue_threshold:
                queue_condition.wait()
        clips = load_and_split_audio(filename, share_mem=True)
        if clips is not None:
            result_queue.put((clips, filename))

def init_workers(dataset, n_workers=NUM_WORKERS):
    """
    Initializes worker processes for multiprocessing, setting up the required queues,
    an event for coordinated exit, and a condition for queue threshold management.

    Parameters:
    - n_workers: Number of worker processes to start.
    - dataset: The dataset items to process.
    - queue_threshold: The threshold for the result queue before producers wait.
    """
    manager = multiprocessing.Manager()
    task_queue = multiprocessing.Queue()
    result_queue = manager.Queue()  # Managed Queue for sharing across processes
    completion_event = manager.Event()
    queue_condition = manager.Condition()
    producers = [
        multiprocessing.Process(
            target=producer,
            args=(task_queue, result_queue, completion_event, queue_condition)
        ) for _ in range(n_workers)
    ]
    for p in producers:
        p.start()
    for item in dataset:
            task_queue.put(item)
    for _ in range(n_workers):
        task_queue.put(None)

    return task_queue, result_queue, producers, completion_event, queue_condition

def process_and_save(batch_audio, batch_meta, specs, model, out_path, conf=False, quiet=False):
    """
    Processes a batch of audio clips and saves the result along with metadata to an HDF5 file.
    """
    stft, band, cqt = specs
    hcqm = compute_hcqm(batch_audio, stft, band, cqt)
    model_device = next(model.parameters()).device
    if not quiet:
        print('hcqm done', hcqm.shape)
    with torch.no_grad():
        hcqm = hcqm.permute(0,3,1,2).to(device=model_device)
        outputs = model(hcqm)
        if not quiet:
            print('model done', outputs.shape)
    torch.cuda.empty_cache()
    results = []
    for meta in batch_meta:
        filename, num_clips, start_idx = meta
        song_outputs = outputs[start_idx:start_idx+num_clips, :]
        probabilities = torch.softmax(song_outputs, dim=1)
        mean_probabilities = probabilities.mean(dim=0)
        confidence_score, predicted_class = torch.max(mean_probabilities, 0)
        predicted_global_bpm = class_to_bpm(predicted_class.item())
        result = {
            "filename": filename,
            "bpm": predicted_global_bpm
        }
        if conf:
            result['confidence'] = confidence_score.item()
        results.append(result)
    with open(out_path, 'a') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

def consume_and_process(result_queue, data_path, queue_condition, n_workers=NUM_WORKERS, max_len_batch=NUM_BATCH, device='cuda', conf=False, quiet=False):
    batch_audio = []
    batch_meta = []
    active_producers = n_workers
    sr = 22050
    len_audio = sr * 8
    if not quiet:
        print(f'Using device: {device}')
    specs = make_kernels(len_audio,  sr, device=device)
    if not quiet:
        print('made kernels')
    model = load_cnn_model(device=device, quiet=quiet)
    model.eval()
    if not quiet:
        print('loaded model')
    total_clips = 0
    if not quiet:
        print(f'producers = {active_producers}')
    while active_producers > 0:
        result = result_queue.get()
        with queue_condition:
            queue_condition.notify_all()
        if result is None:
            active_producers -= 1
            if not quiet:
                print(f'producers = {active_producers}')
            continue
        clips, filename = result
        batch_audio.append(clips)
        num_clips = clips.shape[0]
        start_idx = total_clips
        batch_meta.append((filename, num_clips, start_idx))
        total_clips += num_clips
        if total_clips >= max_len_batch:
            stacked_batch_audio = torch.cat(batch_audio, dim=0).to(device=device)
            process_and_save(stacked_batch_audio, batch_meta, specs,model, data_path, conf=conf, quiet=quiet)
            total_clips = 0
            batch_audio = []
            batch_meta = []

    # Make sure to process any remaining clips
    if batch_audio:
        stacked_batch_audio = torch.cat(batch_audio, dim=0).to(device=device)
        process_and_save(stacked_batch_audio, batch_meta, specs,model, data_path, conf=conf, quiet=quiet)
        pass


def main(dataset, n_workers=NUM_WORKERS, max_len_batch=NUM_BATCH, data_path='output.jsonl', device='cuda', conf=False, quiet=False):
    task_queue, result_queue, producers, completion_event, queue_condition = init_workers(dataset, n_workers)
    try:
        consume_and_process(result_queue, data_path, queue_condition, n_workers=n_workers,max_len_batch=max_len_batch, device=device, conf=conf, quiet=quiet)
    finally:
        completion_event.set()
        for p in producers:
            p.join()  # Ensure all producer processes have finished


def get_audio_files(dir_path):
    """
    Collects all audio files recursively from a specified directory.
    """
    audio_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))
    return audio_files

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    multiprocessing.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Directory containing audio files')
    parser.add_argument('-o', '--output_path', type=str, default='batch_results.jsonl', help='Output path for results')
    parser.add_argument('-d','--device', type=str, default=get_device(), help='Device to use for inference')
    parser.add_argument('-c','--conf', action='store_true', help='Include confidence score in output')
    parser.add_argument('-q','--quiet', action='store_true', help='Use minimal output format')
    args = parser.parse_args()

    songs = get_audio_files(args.input_path)
    if not args.quiet:
        print(len(songs),'songs found')

    start = time.time()
    main(songs, 
         n_workers=NUM_WORKERS, 
         data_path=args.output_path, 
         device=args.device, 
         conf=args.conf,
         quiet=args.quiet
    )
    if not args.quiet:
        print(f'{time.time()-start:.2f}')
    torch.cuda.empty_cache()
