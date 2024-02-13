import zipfile
import os
from data.fma import utils
import pandas as pd


def extract_fma_file(filename):
    path = '/media/bleu/bulkdata/datasets/fma_full.zip'
    extract_path = 'data/'

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extract(filename, extract_path)
    return extract_path+filename

def extract_ballroom_song_bpm(file_path):
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
    dir_path = f'data/BallroomData/{genre}/'
    dircount = len(os.listdir(dir_path))
    extracted_song_bpm = extract_ballroom_song_bpm(file_path)
    results = []
    for (track, media_id,  bpm) in extracted_song_bpm:
        path = dir_path + track + '.wav'
        results.append((path, bpm, genre))
    return results, dircount

def make_ballroom_dataset():
    dirpath = 'data/BallroomData'
    all_entries = os.listdir(dirpath)
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
                all_tracks.append((path, bpm, genre, 'ballroom'))
    return all_tracks

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
            dataset.append((os.path.join(audio_dir, audio_file), bpm, genre, 'giantsteps'))
    return dataset

def make_fma_dataset():
    tracks = utils.load('data/fma/data/fma_metadata/tracks.csv')['track']
    echonest = utils.load('data/fma/data/fma_metadata/echonest.csv')
    genres = utils.load('data/fma/data/fma_metadata/genres.csv')
    AUDIO_DIR = 'data/fma_full/'

    def get_genre(row):
        if row['genres_all']:
            genre_titles = [genres.loc[genre_id, 'title'] for genre_id in row['genres_all'] if genre_id in genres.index]
            genre_full = ''
            for g in genre_titles:
                genre_full += g+', '
            return genre_full[:-2]
        return None

    tracks['genre'] = tracks.apply(get_genre, axis=1)

    tempo_data = echonest['echonest', 'audio_features']['tempo'].astype(int)
    genre_data = tracks['genre'].dropna()

    merged_data = pd.DataFrame({'tempo': tempo_data, 'genre': genre_data}).dropna()

    merged_data['filename'] = merged_data.index.map(lambda track_id: utils.get_audio_path(AUDIO_DIR, track_id))
    merged_data['source'] = 'fma'
    dataset = merged_data.reset_index(drop=True)[['filename', 'tempo', 'genre', 'source']]

    dataset = list(dataset.itertuples(index=False, name=None))
    return dataset