# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License CC-BY-SA 4.0
# You may not use this file except in compliance with the License.

from jiwer import compute_measures
import numpy as np
import jellyfish
import torch
import spacy


class FabricationAnalyzer:
    """
        FabricationAnalyzer class for analyzing lexical and phonetic fabrications in sentences.
        This class uses the jiwer library to compute lexical fabrications and the jellyfish library to compute phonetic fabrications.
    """

    def __init__(
        self, 
        device='cuda'
        ):
        """
            Initialize the FabricationAnalyzer with a device.
            Args:
                device (str): The device to use for computation (e.g., 'cuda' or 'cpu').
        """
        ## Initialize the device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_lexical_fabrications(
        self, 
        reference, 
        hypothesis
        ):
        """
            Compute the lexical fabrications between the reference and hypothesis sentences.
            Args:
                reference (str): The reference sentence.
                hypothesis (str): The hypothesis sentence.
            Returns:
                dict: A dictionary containing the number of insertions, deletions, and substitutions, as well as their ratios.
        """
        ## Check if the reference and hypothesis are the same
        # If they are the same, return 0 for all measures
        if reference == hypothesis:
            return {'insertions_count': 0, 'insertions_ratio': 0.0,
                    'deletions_count': 0, 'deletions_ratio': 0.0,
                    'substitutions_count': 0, 'substitutions_ratio': 0.0}
        ## If the reference is empty and the hypothesis is not, 
        # the number of insertions is the number of words in the hypothesis
        elif len(reference) == 0:
            ins_count = len(hypothesis.split())
            return {'insertions_count': ins_count,
                    'insertions_ratio': 1.0 if ins_count > 0 else 0.0,
                    'deletions_count': 0, 'deletions_ratio': 0.0,
                    'substitutions_count': 0, 'substitutions_ratio': 0.0}
        ## If the hypothesis is empty and the reference is not,
        # the number of deletions is the number of words in the reference
        elif len(hypothesis) == 0:
            del_count = len(reference.split())
            return {'insertions_count': 0, 'insertions_ratio': 0.0,
                    'deletions_count': del_count,
                    'deletions_ratio': 1.0 if del_count > 0 else 0.0,
                    'substitutions_count': 0, 'substitutions_ratio': 0.0}
        ## If the reference and hypothesis are not the same and none of them are empty,
        # compute the measures using the jiwer library
        measures = compute_measures(reference, hypothesis)
        ins = measures['insertions']
        dels = measures['deletions']
        subs = measures['substitutions']
        ## Compute the ratios and return the results
        return {
            'insertions_count': ins,
            'insertions_ratio': ins / len(hypothesis.split()) if len(hypothesis.split()) > 0 else 0,
            'deletions_count': dels,
            'deletions_ratio': dels / len(reference.split()) if len(reference.split()) > 0 else 0,
            'substitutions_count': subs,
            'substitutions_ratio': subs / len(reference.split()) if len(reference.split()) > 0 else 0
            }

    def compute_phonetic_fabrications(
        self, 
        reference, 
        hypothesis
        ):
        """
            Compute the phonetic fabrications between the reference and hypothesis sentences.
            Args:
                reference (str): The reference sentence.
                hypothesis (str): The hypothesis sentence.
            Returns:
                dict: A dictionary containing the phonetic distances between the reference and hypothesis sentences.
        """
        ## Check if the reference and hypothesis are the same
        # If they are the same, return 0 for all measures
        if reference == hypothesis:
            return {
                "hamming": 0.0,
                "levenshtein": 0.0,
                "jaro_winkler": 1.0
                }
        ## Convert the reference and hypothesis to metaphones, i.e., phonetic representations
        ref_meta = jellyfish.metaphone(reference)
        hyp_meta = jellyfish.metaphone(hypothesis)
        ## Compute the hamming phonetic distance using the jellyfish library
        hamm = jellyfish.hamming_distance(ref_meta, hyp_meta)
        max_h = max(len(ref_meta), len(hyp_meta), 1)
        hamm_norm = hamm / max_h if hamm is not None else 0
        ## Compute the levenshtein phonetic distance using the jellyfish library
        leven = jellyfish.levenshtein_distance(ref_meta, hyp_meta)
        max_l = max(len(ref_meta), len(hyp_meta), 1)
        leven_norm = leven / max_l if leven is not None else 0
        ## Compute the jaro_winkler phonetic distance using the jellyfish library
        jaro_winkler = jellyfish.jaro_winkler_similarity(ref_meta, hyp_meta)
        ## Return the results
        return {
            "hamming": hamm_norm,
            "levenshtein": leven_norm,
            "jaro_winkler": jaro_winkler
            }
