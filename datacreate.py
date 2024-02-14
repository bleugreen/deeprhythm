
import sys
sys.path.append('/home/bleu/ai/deeprhythm/src')

import h5py
import os
import torch.multiprocessing as multiprocessing
from deeprhythm.audio_proc.hcqm import make_kernels, compute_hcqm
import torch
import time
from deeprhythm.utils import load_and_split_audio
import csv

NUM_WORKERS = 16
NUM_BATCH = 1024


def producer(task_queue, result_queue, completion_event, queue_condition, queue_threshold=NUM_BATCH):
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
        id, filename, genre,  source,num_clips, bpm = task
        with queue_condition:  # Use the condition to wait if the queue is too full before loading audio
            while result_queue.qsize() >= queue_threshold:
                queue_condition.wait()
        root_dir = '/media/bleu/bulkdata2/deeprhythmdata'
        full_path = os.path.join(root_dir, filename)
        # After ensuring the queue is not full, proceed to load and process audio
        clips = load_and_split_audio(full_path, share_mem=True)
        if clips is not None:
            result_queue.put((clips, filename, bpm, genre, source))

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
    with h5py.File(data_path, 'r') as h5f:
        for item in dataset:
            id, filename, genre, bpm, source, _ = item
            if f'{group}/{os.path.basename(filename)}' not in h5f:
                task_queue.put(item)

    # Signal each producer to terminate once all tasks are processed
    for _ in range(n_workers):
        task_queue.put(None)

    return task_queue, result_queue, producers, completion_event, queue_condition

def process_and_save(batch_audio, batch_meta, specs, h5f_path, group):
    """
    Processes a batch of audio clips and saves the result along with metadata to an HDF5 file.
    """
    # print('batch tensor shape', batch_audio.shape)
    stft, band, cqt = specs
    hcqm = compute_hcqm(batch_audio, stft, band, cqt)
    torch.cuda.empty_cache()
    print('hcqm done', hcqm.shape)
    for meta in batch_meta:
        filename, bpm, genre, source, num_clips, start_idx = meta
        song_clips = hcqm[start_idx:start_idx+num_clips, :, :, :]
        with h5py.File(h5f_path, 'a') as h5f:
            if f'{group}/{os.path.basename(filename)}' in h5f:
                return
            clip_group = h5f.create_group(f'{group}/{os.path.basename(filename)}')
            clip_group.create_dataset('hcqm', data=song_clips.cpu().numpy())
            clip_group.attrs['bpm'] = float(bpm)
            clip_group.attrs['genre'] = genre
            clip_group.attrs['filepath'] = filename
            clip_group.attrs['source'] = source

def consume_and_process(result_queue, data_path, queue_condition, n_workers=NUM_WORKERS, max_len_batch=NUM_BATCH, group='data'):
    batch_audio = []
    batch_meta = []
    active_producers = n_workers
    sr = 22050
    len_audio = sr * 8
    specs = make_kernels(len_audio,  sr)
    total_clips = 0
    print(f'producers = {active_producers}')
    while active_producers > 0:
        result = result_queue.get()
        with queue_condition:
            queue_condition.notify_all()
        if result is None:
            active_producers -= 1
            print(f'producers = {active_producers}')
            continue
        clips, filename, bpm, genre, source = result
        with h5py.File(data_path, 'r') as h5f:
            if f'{group}/{os.path.basename(filename)}' not in h5f:
                batch_audio.append(clips)
                num_clips = clips.shape[0]
                start_idx = total_clips
                batch_meta.append((filename, bpm, genre, source, num_clips, start_idx))
                total_clips += num_clips
        if total_clips >= max_len_batch:
            stacked_batch_audio = torch.cat(batch_audio, dim=0).cuda()
            process_and_save(stacked_batch_audio, batch_meta, specs, data_path, group)
            total_clips = 0
            batch_audio = []
            batch_meta = []

    # Make sure to process any remaining clips
    if batch_audio:
        stacked_batch_audio = torch.cat(batch_audio, dim=0).cuda()
        process_and_save(stacked_batch_audio, batch_meta, specs, data_path, group)
        pass


def main(dataset, n_workers=NUM_WORKERS, max_len_batch=NUM_BATCH, data_path='output.hdf5', group='data'):
    task_queue, result_queue, producers, completion_event, queue_condition = init_workers(dataset,data_path, group, n_workers)
    try:
        consume_and_process(result_queue, data_path, queue_condition, n_workers=n_workers,max_len_batch=max_len_batch, group=group, )
    finally:
        completion_event.set()
        for p in producers:
            p.join()  # Ensure all producer processes have finished


def read_csv_to_tuples(csv_file_path):
    data_tuples = []
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            modified_row = row
            data_tuples.append(tuple(modified_row))
    return data_tuples


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()
    train_songs = read_csv_to_tuples('/media/bleu/bulkdata2/deeprhythmdata/train.csv')
    test_songs = read_csv_to_tuples('/media/bleu/bulkdata2/deeprhythmdata/test.csv')
    val_songs = read_csv_to_tuples('/media/bleu/bulkdata2/deeprhythmdata/val.csv')
    # idx, id, bpm, filename, genre, source
    # print(test_songs[0])
    data_path = '/media/bleu/bulkdata2/deeprhythmdata/hcqm-split.hdf5'
    with h5py.File(data_path, 'w') as hdf5_file:
        # Create groups 'train', 'test', and 'val' within the HDF5 file
        hdf5_file.create_group('train')
        hdf5_file.create_group('test')
        hdf5_file.create_group('val')
    start = time.time()
    main(train_songs, n_workers=16, data_path=data_path, group='train')
    main(test_songs, n_workers=16, data_path=data_path, group='test')
    main(val_songs, n_workers=16, data_path=data_path, group='val')

    print(f'Total Duration: {time.time()-start:.2f}')
    torch.cuda.empty_cache()
    hdf5_filename = data_path
    with h5py.File(hdf5_filename, 'r') as f:
            print('Total Length', sum([len(f.get(key)) for key in f.keys()]))
