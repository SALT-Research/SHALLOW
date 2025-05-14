# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License CC-BY-SA 4.0
# You may not use this file except in compliance with the License.

from whisper_normalizer.english import EnglishTextNormalizer
from jiwer import wer
import argparse
import re


## Define constants for GigaSpeech
PUNCTUATION_TAGS = {
    "<COMMA>": ",",
    "<PERIOD>": ".",
    "<QUESTIONMARK>": "?",
    "<EXCLAMATIONPOINT>": "!"
    }

GARBAGE_TAGS = [
    "<SIL>",
    "<MUSIC>",
    "<NOISE>",
    "<OTHER>"
    ]

FILLERS = [
    'UH', 'UHH', 'UM', 'EH', 'MM', 'HM', 'AH', 'HUH', 'HA', 'ER'
    ]


def parse_args():
    """
        Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Shallow Benchmark")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ls",
        help="Dataset to test"
        )
    parser.add_argument(
        "--model_name",
        type=str,
        default="canary1b",
        help="Model to test"
        )
    parser.add_argument(
        "--gt_transcriptions_path", 
        type=str, 
        default="ls_gt.txt",
        help="Path to the ground truth transcriptions file"
        )
    parser.add_argument(
        "--predictions_path", 
        type=str, 
        default="ls_canary1b.txt", 
        help="Path to the predictions file"
        )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/", 
        help="Path to the output directory"
        )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes to use"
        )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=-1,
        help="Limit the number of examples to process"
        )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
        )
    args = parser.parse_args()
    return args


def clean_transcript_gigaspeech(text, remove_punctuation=False, remove_garbage=True, remove_fillers=True):
    """
        Clean the transcript according to specified parameters
    """
    processed_text = text
    
    # Replace punctuation tags with actual symbols or remove them
    if not remove_punctuation:
        for tag, symbol in PUNCTUATION_TAGS.items():
            # Add a space before the tag if needed for replacement
            processed_text = processed_text.replace(" " + tag, symbol)
            # Also handle cases where the tag is at the beginning or without a space
            processed_text = processed_text.replace(tag + " ", symbol + " ")
            processed_text = processed_text.replace(tag, symbol)
    else:
        for tag in PUNCTUATION_TAGS.keys():
            processed_text = processed_text.replace(tag, "")
    
    # Remove garbage tags
    if remove_garbage:
        for tag in GARBAGE_TAGS:
            processed_text = processed_text.replace(tag, "")
    
    # Remove conversational fillers
    if remove_fillers:
        # Create pattern to match whole words only
        filler_pattern = r'\b(' + '|'.join(FILLERS) + r')\b'
        processed_text = re.sub(filler_pattern, "", processed_text, flags=re.IGNORECASE)
    
    # Clean up any extra whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
    return processed_text


def clean_transcript_models(text):
    """
        Clean the transcript by removing everything within <>
    """
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def safe_wer(gt, hyp):
    """
        Compute WER with special handling for empty strings.
        If both gt and hyp are empty, return 0.0 (perfect match).
        If gt is empty and hyp is not, return 1.0 (full error).
        If gt is not empty and hyp is empty, return 1.0 (full error).
    """
    if not gt.strip():
        if not hyp.strip():
            return 0.0  ## Both empty → perfect match
        else:
            return 1.0  ## GT empty, hyp has content → full error
    return wer(gt, hyp)


def load_transcriptions(transcription_path):
    """
        Load the transcriptions from the file.
        The file should be in the format <segment_id: transcription>
    """
    with open(transcription_path, "r") as f:
        pred_transcriptions = {}
        for line in f:
            line = line.strip()
            line = line.replace("  ", " ")
            if line == "":
                continue
            try:
                segment, transcription = line.split(": ", 1)
                if transcription[0] == "'" or transcription[0] == '"':
                    transcription = transcription[1:]
                if transcription[-1] == "'" or transcription[-1] == '"':
                    transcription = transcription[:-1]
                pred_transcriptions[segment] = transcription
            except:
                segment, transcription = line.split(":")
                pred_transcriptions[segment] = ''
    return pred_transcriptions


def keep_intersection_and_sort(gt_transcriptions, pred_transcriptions):
    """
        Keep only the keys that are in both dictionaries and sort them by key.
    """
    ## Remove the keys that are not in both dictionaries
    keys = set(gt_transcriptions.keys()).intersection(set(pred_transcriptions.keys()))
    gt_transcriptions = {k: gt_transcriptions[k] for k in keys}
    pred_transcriptions = {k: pred_transcriptions[k] for k in keys}
    ## Sort them by key
    gt_transcriptions = dict(sorted(gt_transcriptions.items()))
    pred_transcriptions = dict(sorted(pred_transcriptions.items()))
    return gt_transcriptions, pred_transcriptions


def load_gt_pred_transcriptions(gt_path, predictions_path):
    """
        Load the ground truth and predicted transcriptions.
    """
    ## Load the ground truth and predicted transcriptions
    gt_transcriptions = load_transcriptions(gt_path)
    pred_transcriptions = load_transcriptions(predictions_path)

    ## Keep only the intersection and sort them
    gt_transcriptions, pred_transcriptions = keep_intersection_and_sort(gt_transcriptions, pred_transcriptions)

    ## If the dataset is GigaSpeech, clean the transcriptions
    if "gigaspeech" in gt_path:
        gt_transcriptions = {k: clean_transcript_gigaspeech(v) for k,v in gt_transcriptions.items()}
        pred_transcriptions = {k: clean_transcript_gigaspeech(v) for k,v in pred_transcriptions.items()}

    ## Clean the transcriptions
    gt_transcriptions = {k: clean_transcript_models(str(v)) for k,v in gt_transcriptions.items()}
    pred_transcriptions = {k: clean_transcript_models(str(v)) for k,v in pred_transcriptions.items()}

    ## Normalize the transcriptions
    english_normalizer = EnglishTextNormalizer()
    gt_transcriptions = [english_normalizer(gt) for k,gt in gt_transcriptions.items()]
    pred_transcriptions = [english_normalizer(pt) for k,pt in pred_transcriptions.items()]
    
    return gt_transcriptions, pred_transcriptions