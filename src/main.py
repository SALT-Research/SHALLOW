# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License CC-BY-SA 4.0
# You may not use this file except in compliance with the License.

from functools import partial
import multiprocessing
from tqdm import tqdm
import pandas as pd
import torch
import json
import time
import os

from shallow import ShallowBenchmark
from utils import (
    parse_args,
    load_gt_pred_transcriptions
    )

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("evaluate").setLevel(logging.ERROR)


def init_worker(
    grammar_tool_language,
    language_model,
    local_model_name, 
    global_model_name,
    nli_model_name,
    device_str, 
    worker_id
    ):
    """
        Initialize a worker with its own ShallowBenchmark instance.
    """
    global shallow_bench, worker_identifier
    ## Set the worker ID
    worker_identifier = worker_id
    print(f"Initializing worker {worker_id} with device {device_str}")
    ## Create a dedicated ShallowBenchmark instance for this worker
    shallow_bench = ShallowBenchmark(
        grammar_tool_language=grammar_tool_language,
        language_model=language_model,
        local_model_name=local_model_name,
        global_model_name=global_model_name,
        nli_model_name=nli_model_name,
        device=device_str
        )
    print(f"Worker {worker_id} initialized with its own ShallowBenchmark instance")
    return True

def process_batch(
    batch, 
    verbose=False
    ):
    """
        Process a batch of examples with the worker's pre-initialized ShallowBenchmark.
    """
    global shallow_bench, worker_identifier
    results = []
    error_count = 0
    ## Create a progress bar for this batch
    with tqdm(total=len(batch), desc=f"Worker {worker_identifier}") as pbar:
        for idx, (ref, hyp) in batch:
            start_time = time.time()
            if verbose:
                print(f"Worker {worker_identifier} processing example {idx}")
            ## If reference and hypothesis are the same, use default scores
            if ref == hyp:
                if verbose:
                    print(f"Worker {worker_identifier} found ref==hyp for example {idx}")
                results.append({
                    'ref': ref,
                    'hyp': hyp,
                    'wer': 0,
                    'ins_count': 0,
                    'ins_ratio': 0.0,
                    'del_count': 0,
                    'del_ratio': 0.0,
                    'sub_count': 0,
                    'sub_ratio': 0.0,
                    'phonetic_hamming': 0,
                    'phonetic_levenshtein': 0,
                    'phonetic_jaro_winkler': 0.0,
                    'structural_divergence': 0.0,
                    'gramm_errors': 0,
                    'gramm_errors_spelling': 0,
                    'gramm_errors_grammar': 0,
                    'gramm_errors_punctuation': 0,
                    'local_semantic_window_size_1': 1.0,
                    'local_semantic_window_size_2': 1.0,
                    'local_semantic_window_size_3': 1.0,
                    'global_semantic_cosine_similarity': 1.0,
                    'global_semantic_coherence': 1.0,
                    })
                pbar.update(1)
                continue
            try:
                ## Compute scores using the worker's pre-initialized ShallowBenchmark
                scores = shallow_bench(ref, hyp)
                ## Extract the scores
                wer_score = scores['wer_score']
                lexical_score = scores['lexical_fabrication_score']
                phonetic_score = scores['phonetic_fabrication_score']
                morph_score = scores['morphological_hallucination_score']
                local_semantic_score = scores['local_semantic_score']
                global_semantic_score = scores['global_semantic_score']
                ## Append the results to the list
                results.append({
                    'ref': ref,
                    'hyp': hyp,
                    'wer': scores['wer_score'],
                    'ins_count': lexical_score['insertions_count'],
                    'ins_ratio': lexical_score['insertions_ratio'],
                    'del_count': lexical_score['deletions_count'],
                    'del_ratio': lexical_score['deletions_ratio'],
                    'sub_count': lexical_score['substitutions_count'],
                    'sub_ratio': lexical_score['substitutions_ratio'],
                    'phonetic_hamming': phonetic_score['hamming'],
                    'phonetic_levenshtein': phonetic_score['levenshtein'],
                    'phonetic_jaro_winkler': phonetic_score['jaro_winkler'],
                    'structural_divergence': morph_score['structural_divergence'],
                    'gramm_errors': morph_score['grammatical_errors']['total_errors'],
                    'gramm_errors_spelling': morph_score['grammatical_errors']['error_categories']['spelling'],
                    'gramm_errors_grammar': morph_score['grammatical_errors']['error_categories']['grammar'],
                    'gramm_errors_punctuation': morph_score['grammatical_errors']['error_categories']['punctuation'],
                    'local_semantic_window_size_1': local_semantic_score['window_size_1'],
                    'local_semantic_window_size_2': local_semantic_score['window_size_2'],
                    'local_semantic_window_size_3': local_semantic_score['window_size_3'],
                    'global_semantic_cosine_similarity': global_semantic_score['global_semantic_cosine_similarity'],
                    'global_semantic_coherence': global_semantic_score['global_semantic_coherence'],
                    })
                if verbose:
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"Worker {worker_identifier} completed example {idx} in {duration:.2f} seconds")
            ## Handle exceptions and log errors
            except Exception as e:
                error_count += 1
                print(f"Worker {worker_identifier} error on example {idx}: {str(e)}")
                print(f"Reference: {ref}")
                print(f"Hypothesis: {hyp}")
            ## Update the progress bar
            pbar.update(1)
    ## Return the results and error count
    return results, error_count

def chunk_data(
    data, 
    num_chunks
    ):
    """
        Split data into approximately equal chunks.
    """
    ## Calculate the size of each chunk
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    ## Create a list to hold the chunks
    chunks = []
    start = 0
    ## Split the data into chunks
    for i in range(num_chunks):
        ## Add one extra item to the first 'remainder' chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start:end])
        start = end
    ## Return the list of chunks
    return chunks

def main():
    """
        Main function to run the Shallow Benchmark.
    """
    ## Set multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    ## Parse arguments
    args = parse_args()

    ## Load the transcriptions
    gt_transcriptions, pred_transcriptions = load_gt_pred_transcriptions(
        gt_path=args.gt_transcriptions_path,
        predictions_path=args.predictions_path
        )

    ## Limit the number of examples to process
    if args.examples_limit > 0:
        gt_transcriptions = gt_transcriptions[:args.examples_limit]
        pred_transcriptions = pred_transcriptions[:args.examples_limit]
    print(f"Loaded {len(gt_transcriptions)} examples")
    print(f"Loaded {len(pred_transcriptions)} examples")

    ## Create a list of indexed examples
    examples = list(enumerate(zip(gt_transcriptions, pred_transcriptions)))

    ## Determine the number of workers to use
    num_workers = min(args.num_workers, len(examples))
    if torch.cuda.is_available():
        device_str = 'cuda'
        ## Limit workers if using GPU to prevent memory issues
        num_workers = min(num_workers, 4)
    else:
        device_str = 'cpu'

    ## Print the configuration
    print('Dataset:', args.dataset_name)
    print('Model:', args.model_name)
    print(f"Processing {len(examples)} examples with {num_workers} workers on {device_str}")

    ## Make sure the output directory exists
    os.makedirs(f'{args.output_dir}', exist_ok=True)

    ## Split examples into batches for each worker
    batches = chunk_data(examples, num_workers)

    ## Initialize parameters for ShallowBenchmark
    init_params = {     
        'grammar_tool_language': 'en-US',
        'language_model': 'en_core_web_sm',
        'local_model_name': 'bert-base-uncased',
        'global_model_name': 'nli-roberta-base-v2',
        'nli_model_name': 'facebook/bart-large-mnli',
        'device_str': device_str
        }
        
    ## Create a pool with pre-initialized workers
    all_results = []
    total_errors = 0
    print("Starting worker pool with pre-initialized ShallowBenchmark instances")
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(
            init_params['grammar_tool_language'],
            init_params['language_model'],
            init_params['local_model_name'],
            init_params['global_model_name'],
            init_params['nli_model_name'],
            init_params['device_str'],
            ## Pass worker ID as the last argument
            multiprocessing.Value('i', 0)
            )
        ) as pool:
        ## Process each batch with a dedicated worker
        process_func = partial(process_batch, verbose=args.verbose)
        results = pool.map(process_func, batches)
        ## Collect all results
        for batch_results, batch_errors in results:
            all_results.extend(batch_results)
            total_errors += batch_errors
    print(f"Parallel processing complete. Total errors: {total_errors}")
    ## Check if any results were processed unsuccessfully
    if not all_results:
        print("No results were processed successfully. Exiting.")
        return

    ## Convert results to DataFrame
    df = pd.DataFrame(all_results)

    ## Save partial results
    print("Saving partial results")
    df.to_csv(f"{args.output_dir}/shallow_metrics_{args.dataset_name}_{args.model_name}_partial.csv", index=False)

    ## Create a new ShallowBenchmark instance for aggregated calculations
    print("Creating ShallowBenchmark for aggregated calculations")
    shallow_bench = ShallowBenchmark(
        grammar_tool_language='en-US',
        language_model='en_core_web_sm',
        local_model_name='bert-base-uncased', 
        global_model_name='nli-roberta-base-v2',
        nli_model_name='facebook/bart-large-mnli',
        device=device_str
        )
    
    ## Calculate the WER for the entire dataset
    print("Calculating WER for the entire dataset")
    wer_score = shallow_bench.compute_dataset_wer(
        gt_transcriptions,
        pred_transcriptions
        )

    ## Calculate the lexical fabrication scores
    print("Calculating lexical fabrication scores")
    lexical_fabrication_scores = shallow_bench.aggregated_lexical_fabrication_score(
        ins_ratios=df['ins_ratio'].tolist(),
        del_ratios=df['del_ratio'].tolist(),
        sub_ratios=df['sub_ratio'].tolist(),
        hypotheses=pred_transcriptions
        )
    df['lexical_fabrication_score'] = lexical_fabrication_scores

    ## Calculate the phonetic fabrication scores
    print("Calculating phonetic fabrication scores")
    phonetic_fabrication_scores = shallow_bench.aggregated_phonetic_score(
        hammings=df['phonetic_hamming'].tolist(),
        levenshteins=df['phonetic_levenshtein'].tolist(),
        jaro_winklers=df['phonetic_jaro_winkler'].tolist(),
        )
    df['phonetic_fabrication_score'] = phonetic_fabrication_scores
    
    ## Calculate the morphological hallucination scores
    print("Calculating morphological hallucination scores")
    structural_divergence_scores, grammatical_errors_scores, morphological_hallucination_scores = shallow_bench.aggregated_morphological_hallucination_score(
        syntax_divergences=df['structural_divergence'].tolist(),
        spelling_errors=df['gramm_errors_spelling'].tolist(),
        grammar_errors=df['gramm_errors_grammar'].tolist(),
        punctuation_errors=df['gramm_errors_punctuation'].tolist(),
        hypotheses=pred_transcriptions
        )
    df['structural_divergence_score'] = structural_divergence_scores
    df['grammatical_errors_score'] = grammatical_errors_scores
    df['morphological_hallucination_score'] = morphological_hallucination_scores

    ## Calculate the contextual hallucination scores
    print("Calculating contextual hallucination scores")
    local_semantic_scores = shallow_bench.aggregated_local_semantic_score(
        c1s=df['local_semantic_window_size_1'].tolist(),
        c2s=df['local_semantic_window_size_2'].tolist(),
        c3s=df['local_semantic_window_size_3'].tolist(), 
        )
    df['local_semantic_score'] = local_semantic_scores

    ## Calculate the global semantic scores
    print("Calculating global semantic scores")
    global_semantic_scores = shallow_bench.aggregated_global_semantic_score(
        cosines=df['global_semantic_cosine_similarity'].tolist(),
        semantic_coherences=df['global_semantic_coherence'].tolist(),
        )
    df['global_semantic_score'] = global_semantic_scores

    ## Calculate the semantic hallucination scores
    print("Calculating semantic hallucination scores")
    semantic_score = shallow_bench.aggregated_semantic_score(
        local_semantic_scores=df['local_semantic_score'].tolist(),
        global_semantic_scores=df['global_semantic_score'].tolist()
        )
    df['semantic_hallucination_score'] = semantic_score

    ## Calculate the dataset statistics
    print("Calculating dataset statistics")
    stats = shallow_bench.compute_dataset_stats(df)

    ## Save the results
    print("Saving results")
    data = {
        'wer_score': wer_score,
        'lexical_fabrication_score': stats['lexical_fabrication_score'],
        'phonetic_fabrication_score': stats['phonetic_fabrication_score'],
        'morphological_hallucination_score': stats['morphological_hallucination_score'],
        'semantic_hallucination_score': stats['semantic_hallucination_score'], 
        }
    with open(f"{args.output_dir}/shallow_stats_{args.dataset_name}_{args.model_name}.json", 'w') as f:
        json.dump(data, f, indent=4)
    
    ## Save the DataFrame to a CSV file
    print("Saving complete dataset to CSV")
    df.to_csv(f"{args.output_dir}/shallow_metrics_{args.dataset_name}_{args.model_name}.csv", index=False)


if __name__ == "__main__":
    main()