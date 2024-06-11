import argparse
import warnings

from deeprhythm.model.predictor import DeepRhythmPredictor
from deeprhythm.utils import get_device


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Path to the audio file to analyze')
    parser.add_argument('-d','--device', type=str, default=get_device(), help='Device to use for inference')
    parser.add_argument('-c','--conf', action='store_true', help='Include confidence score in output')
    parser.add_argument('-q','--quiet', action='store_true', help='Use minimal output format')
    args = parser.parse_args()


    predictor = DeepRhythmPredictor(device=args.device, quiet=args.quiet)
    result = predictor.predict(args.filename, include_confidence=args.conf)
    if args.conf:
        bpm, conf = result
    else:
        bpm = result
    
    if args.quiet:
        print(result)
    else:
        print(f'Predicted BPM: {bpm}')
        if args.conf:
            print(f'Confidence: {conf}')