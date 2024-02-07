
import librosa
import numpy as np
import h5py
import os
import torch.multiprocessing as multiprocessing
from hcqm import make_specs, compute_hcqm
import torch
import time
from data.fma import utils
import pandas as pd
# Ensure torch.multiprocessing is set up correctly for sharing CUDA tensors if needed
import ffmpeg

def read_mp3(file_path, sample_rate=44100):
    """
    Reads an MP3 file and returns the audio data as a numpy array.

    Parameters:
    - file_path: Path to the MP3 file.
    - sample_rate: Desired sample rate (default is 44100 Hz).

    Returns:
    - A numpy array containing the audio data.
    """
    # Run FFmpeg process and output raw audio data (PCM) to stdout
    out, _ = (
        ffmpeg
        .input(file_path)
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=sample_rate, af='pan=mono|c0=.5*c0+.5*c1')
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Convert the bytes read from stdout into a numpy array
    audio_array = np.frombuffer(out, np.int16)

    return audio_array

def extract_song_bpm(file_path):
    """
    Extracts (song_id, bpm) tuples from the given metadata file.

    Args:
    - file_path (str): The path to the metadata file.

    Returns:
    - List[Tuple[str, str]]: A list of tuples containing the song ID and BPM.
    """
    song_bpm_pairs = []
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('http://www.ballroomdancers.com/Music/'):
            # Extract song ID
            parts = line.split('/')
            if 'Media' in parts:
                song_name = 'Media-' + parts[-1].split('.')[0]
            else:
                song_name = parts[-3] + '-' + parts[-2] + '-' + parts[-1].split('.')[0]
            song_name = song_name.replace('.ram', '.wav')

            # Look for BPM in the subsequent lines
            i += 1
            song_id = 'Media-' +lines[i].split('Song=')[1].split(' ')[0]
            while i < len(lines) and ' BPM' not in lines[i]:
                i += 1
            if i < len(lines):
                bpm = lines[i].strip().split(' ')[0]
                song_bpm_pairs.append((song_name, song_id, bpm))
        i += 1

    return song_bpm_pairs

def process_ballroom_folder(genre):
    file_path = f'data/BallroomData/nada/{genre}.log'
    dir_path = f'/data/BallroomData/{genre}/'
    dircount = len(os.listdir(dir_path))
    extracted_song_bpm = extract_song_bpm(file_path)
    results = []
    for (track, media_id,  bpm) in extracted_song_bpm:
        path = dir_path + track + '.wav'
        results.append((path, bpm, genre))
    return results, dircount

def make_ballroom_dataset():
    dirpath = 'data/BallroomData'
    all_entries = os.listdir(dirpath)
    # Filter directories that start with a capital letter
    genres = [entry for entry in all_entries if os.path.isdir(os.path.join(dirpath, entry)) and entry[0].isupper()]

    invalid = []
    all_tracks = []
    track_count = 0
    for genre in genres:
        tracks, dircount = process_ballroom_folder(genre)
        track_count += dircount
        for (path, bpm, genre) in tracks:
            if not os.path.isfile(path):
                invalid.append((path, bpm, genre))
            else:
                all_tracks.append((path, bpm, genre))
    return all_tracks[:10]

def make_giantsteps_dataset():
    audio_dir = 'data/giantsteps-tempo-dataset/audio'
    tempo_dir = 'data/giantsteps-tempo-dataset/annotations_v2/tempo'
    genre_dir = 'data/giantsteps-tempo-dataset/annotations/genre'
    audio_files = os.listdir(audio_dir)
    dataset = []
    for audio_file in audio_files:
        if audio_file.endswith('.mp3'):
            audio_id = audio_file[:-4]
            bpm_path = os.path.join(tempo_dir, f'{audio_id}.bpm')
            genre_path = os.path.join(genre_dir, f'{audio_id}.genre')
            with open(bpm_path, 'r') as bpm_file:
                bpm = float(bpm_file.read().strip())
            with open(genre_path, 'r') as genre_file:
                genre = genre_file.read().strip()
            dataset.append((os.path.join(audio_dir, audio_file), bpm, genre))
    return dataset[:10]

def make_fma_dataset():
    tracks = utils.load('data/fma/data/fma_metadata/tracks.csv')['track']
    echonest = utils.load('data/fma/data/fma_metadata/echonest.csv')
    genres = utils.load('data/fma/data/fma_metadata/genres.csv')
    AUDIO_DIR = '/mnt/datasets/fma_full'

    def get_genre(row):
        if row['genres_all']:
            # Retrieve the title of each genre in genres_all
            genre_titles = [genres.loc[genre_id, 'title'] for genre_id in row['genres_all'] if genre_id in genres.index]
            genre_full = ''
            for g in genre_titles:
                genre_full += g+', '
            return genre_full[:-2]
        return None

    tracks['genre'] = tracks.apply(get_genre, axis=1)

    # Directly creating a DataFrame from 'tempo_data' and 'genre'
    tempo_data = echonest['echonest', 'audio_features']['tempo'].astype(int)  # Ensuring tempo is integer
    genre_data = tracks['genre'].dropna()

    # Merging on index to align track IDs, ensuring both tempo and genre data match per track ID
    merged_data = pd.DataFrame({'tempo': tempo_data, 'genre': genre_data}).dropna()

    # Vectorizing the filename construction to avoid explicit looping
    merged_data['filename'] = merged_data.index.map(lambda track_id: utils.get_audio_path(AUDIO_DIR, track_id))

    # Resetting index to ensure dataset is clean and index-independent
    dataset = merged_data.reset_index(drop=True)[['filename', 'tempo', 'genre']]

    dataset = list(dataset.itertuples(index=False, name=None))
    return dataset

def load_and_split_audio(filename, sr=22050, clip_length=8):
    """
    Load an audio file, split it into 8-second clips, and return a single tensor of all clips.

    Parameters:
    - filename: Path to the audio file.
    - sr: Sampling rate to use for loading the audio.
    - clip_length: Length of each clip in seconds.

    Returns:
    A tensor of shape [clips, audio] where each row is an 8-second clip.
    """

    clips = []
    clip_samples = sr * clip_length
    try:
        if filename.endswith('.mp3'):
            audio = read_mp3(filename, sr)
        else:
            audio, _ = librosa.load(filename, sr=sr)
        for i in range(0, len(audio), clip_samples):
            if i + clip_samples <= len(audio):
                clip_tensor = torch.tensor(audio[i:i + clip_samples], dtype=torch.float32)
                clips.append(clip_tensor)
    except Exception as e:
        print(e, filename)

    # Stack all clips along a new dimension to form a single tensor
    if clips:
        stacked_clips = torch.stack(clips, dim=0)
    else:
        # Return an empty tensor if no clips were created (file is shorter than clip_length)
        return None

    # Share memory of the stacked clips tensor
    stacked_clips.share_memory_()

    return stacked_clips


def producer(task_queue, result_queue, completion_event):
    """
    Producer function for multiprocessing: loads and preprocesses audio files,
    then puts the results into the result queue. Waits on a shared Event before exiting.

    Parameters:
    - task_queue: Queue from which tasks are retrieved.
    - result_queue: Queue where results are put.
    - completion_event: An Event object that signals when the producer can safely exit.
    """
    while True:
        task = task_queue.get()
        if task is None:
            result_queue.put(None)  # Send termination signal to indicate this producer is done
            completion_event.wait()  # Wait for the signal to exit
            break
        filename, bpm, genre = task
        clips = load_and_split_audio(filename)
        if clips is not None:
            result_queue.put((clips, filename, bpm, genre))

def init_workers(n_workers, dataset):
    """
    Initializes worker processes for multiprocessing and an Event for coordinated exit.
    """
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    completion_event = multiprocessing.Event()  # Create a shared Event

    # Pass the shared Event to each producer
    producers = [multiprocessing.Process(target=producer, args=(task_queue, result_queue, completion_event)) for _ in range(n_workers)]
    for p in producers:
        p.start()

    for item in dataset:
        task_queue.put(item)
    for _ in range(n_workers):
        task_queue.put(None)  # Send termination signal to each producer

    return task_queue, result_queue, producers, completion_event

def process_and_save(batch_audio, batch_meta, specs, h5f_path, group_name='data'):
    """
    Processes a batch of audio clips and saves the result along with metadata to an HDF5 file.
    """
    # Convert batch_audio to a tensor
    batch_tensor = batch_audio

    print('batch tensor shape', batch_tensor.shape)
    # Compute hcqm
    stft, band, cqt = specs
    hcqm = compute_hcqm(batch_tensor, stft, band, cqt)  # Assume this returns a tensor of shape [len_batch, f, b, h]
    print('hcqm complete', hcqm.shape)
    torch.cuda.empty_cache()
    # Ensure thread-safe writing to the HDF5 file
    with multiprocessing.Lock():
        with h5py.File(h5f_path, 'a') as h5f:
            for i, (hcqm_tensor, meta) in enumerate(zip(hcqm, batch_meta)):
                filename, bpm, genre = meta
                # Create a unique group for each clip to avoid collisions
                clip_group = h5f.create_group(f'{group_name}/{os.path.basename(filename)}_{i}')
                # Save hcqm data
                clip_group.create_dataset('hcqm', data=hcqm_tensor.cpu().numpy())
                # Save metadata
                clip_group.attrs['bpm'] = bpm
                clip_group.attrs['genre'] = genre
                clip_group.attrs['filepath'] = filename

def consume_and_process(result_queue, data_path, n_workers=32, max_len_batch=2048, group='data'):
    batch_audio = []
    batch_meta = []
    active_producers = n_workers
    sr = 22050
    len_audio = sr * 8
    specs = make_specs(len_audio,  sr)
    total_clips = 0
    print(f'producers = {active_producers}')
    while active_producers > 0:

        result = result_queue.get()
        if result is None:
            active_producers -= 1
            print(f'producers = {active_producers}')
            continue  # Skip processing and wait for more data or other sentinels

        # Process your data here
        clips, filename, bpm, genre = result
        batch_audio.append(clips)
        total_clips += clips.shape[0]

        batch_meta.extend([(filename, bpm, genre)] * clips.shape[0])
        # print('num clips=', sum(clip.shape[0] for clip in batch_audio))
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


def main(dataset, n_workers=64, max_len_batch=2048, data_path='output.hdf5', group='data'):
    task_queue, result_queue, producers, completion_event = init_workers(n_workers, dataset)
    try:
        consume_and_process(result_queue, data_path, n_workers=n_workers,max_len_batch=max_len_batch, group=group)
    finally:
        completion_event.set()
        for p in producers:
            p.join()  # Ensure all producer processes have finished


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()
    fma_dataset = make_fma_dataset()
    # gs_dataset = make_giantsteps_dataset()
    br_dataset = make_ballroom_dataset()

    # # dataset = fma_dataset + gs_dataset + br_dataset
    # # print(f'total len = {len(dataset)}')


    start = time.time()
    main(fma_dataset, n_workers=16, data_path='output_data_nogs.hdf5', group='fma')
    print(f'FMA Duration: {time.time()-start:.2f}')
    # gs_start = time.time()
    # main(gs_dataset, n_workers=64, data_path='output_data_mini.hdf5', group='giantsteps')
    # print(f'GS Duration: {time.time()-gs_start:.2f}')

    br_start = time.time()
    main(br_dataset, n_workers=64, data_path='output_data_nogs.hdf5', group='ballroom')
    print(f'BR Duration: {time.time()-br_start:.2f}')

    print(f'Total Duration: {time.time()-start:.2f}')
    torch.cuda.empty_cache()
    hdf5_filename = 'output_data_nogs.hdf5'
    with h5py.File(hdf5_filename, 'r') as f:
            # Access the group for the song
            print(len(f.get('ballroom'))+len(f.get('fma')))