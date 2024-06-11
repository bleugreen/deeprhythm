
import sys
sys.path.append('../deeprhythm/src')
import json

import os
import torch.multiprocessing as multiprocessing
import torch
import time
from deeprhythm.utils import load_and_split_audio
from deeprhythm.audio_proc.hcqm import make_kernels, compute_hcqm
from deeprhythm.model.infer import load_cnn_model
from deeprhythm.utils import class_to_bpm
from deeprhythm.utils import get_device
import librosa

NUM_WORKERS = 8
NUM_BATCH = 256


def producer(task_queue, result_queue, completion_event, queue_condition, queue_threshold=NUM_BATCH*2):
    """
    Producer function that waits on a shared condition if the result_queue is above a certain threshold
    immediately after getting a task and before loading and processing the audio.
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
        # After ensuring the queue is not full, proceed to load and process audio
        y, sr = librosa.load(filename, sr=22050)
        bpm = librosa.beat.tempo(y=y, sr=sr)
        if bpm:
            result_queue.put((bpm, filename))


def init_workers(dataset, data_path, group, n_workers=NUM_WORKERS):
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

    # Create producer processes
    producers = [
        multiprocessing.Process(
            target=producer,
            args=(task_queue, result_queue, completion_event, queue_condition)
        ) for _ in range(n_workers)
    ]

    # Start all producers
    for p in producers:
        p.start()

    for item in dataset:
            task_queue.put(item)

    # Signal each producer to terminate once all tasks are processed
    for _ in range(n_workers):
        task_queue.put(None)

    return task_queue, result_queue, producers, completion_event, queue_condition



def consume_and_process(result_queue, data_path, queue_condition, n_workers=NUM_WORKERS, max_len_batch=NUM_BATCH, device='cuda'):

    active_producers = n_workers
    print(f'producers = {active_producers}')
    while active_producers > 0:
        result = result_queue.get()
        with queue_condition:
            queue_condition.notify_all()
        if result is None:
            active_producers -= 1
            print(f'producers = {active_producers}')
            continue
        bpm, filename = result
        print(f'filename: {filename}, bpm: {bpm}')

def main(dataset, n_workers=NUM_WORKERS, max_len_batch=NUM_BATCH, data_path='output.hdf5', device='cuda'):
    task_queue, result_queue, producers, completion_event, queue_condition = init_workers(dataset,data_path, 'group', n_workers)
    try:
        consume_and_process(result_queue, data_path, queue_condition, n_workers=n_workers,max_len_batch=max_len_batch, device=device)
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
    multiprocessing.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()

    root_dir = sys.argv[1]
    songs = get_audio_files(root_dir)
    print(len(songs),'songs found')
    data_path = sys.argv[2] if len(sys.argv) > 2 else 'batch_results.jsonl'

    start = time.time()
    main(songs, n_workers=NUM_WORKERS, data_path=data_path)

    print(f'Total Duration: {time.time()-start:.2f}')
    torch.cuda.empty_cache()
    with open(data_path, 'r') as f:
        print('Total Length', sum(1 for _ in f))
