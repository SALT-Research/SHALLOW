# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License CC-BY-SA 4.0
# You may not use this file except in compliance with the License.

from jiwer import wer
import numpy as np
import benepar
import torch
import nltk

from semantic import SemanticAnalyzer
from fabrications import FabricationAnalyzer
from morphological import MorphologicalAnalyzer


class ShallowBenchmark:
    """
        ShallowBenchmark class for analyzing hallucinations in ASR systems.
        This class uses various analyzers to compute different types of hallucination scores.
        The analyzers include FabricationAnalyzer, MorphologicalAnalyzer, and SemanticAnalyzer.
        The class also provides methods to compute aggregated scores for different types of hallucinations.
    """

    def __init__(
        self,
        grammar_tool_language='en-US',
        language_model='en_core_web_sm',
        local_model_name='bert-base-uncased', 
        global_model_name='nli-roberta-base-v2',
        nli_model_name='facebook/bart-large-mnli',
        device='cuda' if torch.cuda.is_available() else 'cpu'
        ):
        """
            Initialize the ShallowBenchmark with the necessary analyzers.
            Args:
                grammar_tool_language (str): The language for the grammar checker.
                language_model (str): The language model for parsing sentences.
                local_model_name (str): The local model name for semantic analysis.
                global_model_name (str): The global model name for semantic analysis.
                nli_model_name (str): The NLI model name for semantic analysis.
                device (str): The device to use for computation (e.g., 'cuda' or 'cpu').
        """
        ## Download the necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        ## Download the necessary Benepar resources
        benepar.download('benepar_en3')
        ## Initialize the analyzers
        ### FabricationAnalyzer
        self.fabrications = FabricationAnalyzer(
            device=device
            )
        ### MorphologicalAnalyzer
        self.morphological = MorphologicalAnalyzer(
            grammar_tool_language=grammar_tool_language,
            language_model=language_model
            )
        ### SemanticAnalyzer
        self.semantic = SemanticAnalyzer(
            local_model_name=local_model_name,
            global_model_name=global_model_name,
            nli_model_name=nli_model_name,
            device=device
            )

    def __call__(
        self, 
        ref, 
        hyp
        ):
        """
            Compute the hallucination scores for the given reference and hypothesis sentences.
            Args:
                ref (str): The reference sentence.
                hyp (str): The hypothesis sentence.
            Returns:
                dict: A dictionary containing the hallucination scores.
        """
        ## Compute the WER score
        wer_score = self.safe_wer(ref, hyp)
        ## Compute the texical fabrication score
        lexical_fabr_score = self.fabrications.compute_lexical_fabrications(ref, hyp)
        ## Compute the phonetic fabrication score
        phonetic_fabr_score = self.fabrications.compute_phonetic_fabrications(ref, hyp)
        ## Compute the morphological hallucination score
        morph_score = self.morphological.morphological_hallucination_score(ref, hyp)
        ## Compute the local semantic score
        local_semantic_score = self.semantic.local_semantic_score(ref, hyp)
        ## Compute the global semantic score
        global_semantic_score = self.semantic.global_semantic_score(ref, hyp)
        return {
            'wer_score': wer_score,
            'lexical_fabrication_score': lexical_fabr_score,
            'phonetic_fabrication_score': phonetic_fabr_score,
            'morphological_hallucination_score': morph_score,
            'local_semantic_score': local_semantic_score,
            'global_semantic_score': global_semantic_score
            }

    def compute_dataset_wer(
        self,
        gt_transcriptions, 
        pred_transcriptions
        ):
        """
            Compute the Word Error Rate (WER) for a dataset of transcriptions.
            Args:
                gt_transcriptions (list): List of ground truth transcriptions.
                pred_transcriptions (list): List of predicted transcriptions.
            Returns:
                float: The WER score for the dataset.
        """
        ## Compute the WER score for the dataset
        corpus_gt = " ".join(gt_transcriptions)
        corpus_pred = " ".join(pred_transcriptions)
        wer_score = round(100*wer(corpus_gt, corpus_pred), 2)
        return wer_score

    # @staticmethod
    def safe_wer(
        self,
        reference, 
        hypothesis
        ):
        """
            Compute the Word Error Rate (WER) between the reference and hypothesis sentences.
            Args:
                reference (str): The reference sentence.
                hypothesis (str): The hypothesis sentence.
            Returns:
                float: The WER score.
        """
        ## If the reference is none or empty, return 0.0 if the hypothesis is also empty, else return 1.0
        if not reference.strip():
            return 0.0 if not hypothesis.strip() else 1.0
        ## If neither the reference nor the hypothesis are empty, compute the WER using the jiwer library
        return wer(reference, hypothesis)

    # @staticmethod
    def aggregated_lexical_fabrication_score(
        self,
        ins_ratios, 
        del_ratios, 
        sub_ratios, 
        hypotheses,
        fillers=['actually', 'literally', 'definitely', 'er', 'just', 'absolutely', 'mmm', 'ah', 'well', 
            'seriously', 'basically', 'you know', 'I mean', 'ahm', 'like', 'sort of', 
            'I guess', 'I suppose', 'I think', 'right', 'ok', 'um', 'mm', 'probably', 'totally', 
            'kind of', 'uh', 'maybe', 'no doubt', 'okay', 'uhm', 'really', 'so', 'for sure'
            ]
        ):
        """
            Compute the aggregated lexical fabrication score based on insertion, deletion, and substitution ratios.
            Args:
                ins_ratios (list): List of insertion ratios.
                del_ratios (list): List of deletion ratios.
                sub_ratios (list): List of substitution ratios.
                hypotheses (list): List of hypothesis sentences.
                fillers (list): List of filler words.
            Returns:
                list: List of aggregated lexical fabrication scores.
        """
        ## Compute the aggregated lexical fabrication score
        scores = []
        for ins, dele, sub, hyp in zip(ins_ratios, del_ratios, sub_ratios, hypotheses):
            if ins == 1 and all(word in fillers for word in hyp.split()):
                scores.append(1.0)
            else:
                scores.append(0.5 * ins + 0.3 * sub + 0.2 * dele)
        return scores

    # @staticmethod
    def aggregated_phonetic_score(
        self,
        hammings, 
        levenshteins, 
        jaro_winklers
        ):
        """
            Compute the aggregated phonetic score based on Hamming, Levenshtein, and Jaro-Winkler distances.
            Args:
                hammings (list): List of Hamming distances.
                levenshteins (list): List of Levenshtein distances.
                jaro_winklers (list): List of Jaro-Winkler distances.
            Returns:
                list: List of aggregated phonetic scores.
        """
        ## Compute the aggregated phonetic score
        ### The score is computed as a weighted average of the distances,
        ### with each distance having a weight of 1/3
        scores = []
        for h, l, j in zip(hammings, levenshteins, jaro_winklers):
            ## Convert the jaro-winkler similarity to a distance
            j_dist = 1 - j
            scores.append((h + l + j_dist) / 3)
        return scores

    def aggregated_grammatical_errors_score(
        self,
        spelling_errors, 
        grammar_errors, 
        punctuation_errors, 
        hypotheses
        ):
        """
            Compute the aggregated grammatical errors score based on spelling, grammar, and punctuation errors.
            Args:
                spelling_errors (list): List of spelling errors.
                grammar_errors (list): List of grammar errors.
                punctuation_errors (list): List of punctuation errors.
                hypotheses (list): List of hypothesis sentences.
            Returns:
                list: List of aggregated grammatical errors scores.
        """
        ## Compute the aggregated grammatical errors score
        ### The score is computed as a weighted average of the errors
        ### The weights are 0.4 for grammar errors, 0.4 for spelling errors, and 0.2 for punctuation errors
        ### The score is normalized by the number of words in the hypothesis
        scores = []
        for s, g, p, hyp in zip(spelling_errors, grammar_errors, punctuation_errors, hypotheses):
            try:
                n = len(hyp.split())
                score = (0.4 * g + 0.4 * s + 0.2 * p) / n if n > 0 else 0
            except:
                score = 0
            scores.append(score)
        return scores

    # @staticmethod
    def aggregated_morphological_hallucination_score(
        self,
        syntax_divergences, 
        spelling_errors, 
        grammar_errors, 
        punctuation_errors, 
        hypotheses
        ):
        """
            Compute the aggregated morphological hallucination score based on syntax divergences and grammatical errors.
            Args:
                syntax_divergences (list): List of syntax divergences.
                spelling_errors (list): List of spelling errors.
                grammar_errors (list): List of grammar errors.
                punctuation_errors (list): List of punctuation errors.
                hypotheses (list): List of hypothesis sentences.
            Returns:
                tuple: Tuple containing the syntax divergences, grammatical errors, and aggregated morphological hallucination scores.
        """
        ## Compute the aggregated morphological hallucination score
        ### The score is computed as a weighted average of the syntax divergences and grammatical errors
        ### The weights are 0.4 for syntax divergences and 0.6 for grammatical errors
        st = syntax_divergences
        ge = self.aggregated_grammatical_errors_score(spelling_errors, grammar_errors, punctuation_errors, hypotheses)
        return st, ge, [0.4 * s + 0.6 * g for s, g in zip(st, ge)]

    # @staticmethod
    def aggregated_local_semantic_score(
        self,
        c1s, 
        c2s, 
        c3s
        ):
        """
            Compute the aggregated local semantic score based on scores from different window sizes.
            Args:
                c1s (list): List of scores from window size equal to 1.
                c2s (list): List of scores from window size equal to 2.
                c3s (list): List of scores from window size equal to 3.
            Returns:
                list: List of aggregated local semantic scores.
        """
        ## Compute the aggregated local semantic score
        ### The score is computed as a weighted average of the scores from different window sizes
        ### The weights are 0.5 for window size 1, 0.3 for window size 2, and 0.2 for window size 3
        return [0.5 * (1 - c1) + 0.3 * (1 - c2) + 0.2 * (1 - c3) for c1, c2, c3 in zip(c1s, c2s, c3s)]

    def aggregated_semantic_distance_score(
        self,
        cosines
        ):
        """
            Compute the aggregated semantic distance score based on cosine distances.
            Args:
                cosines (list): List of cosine distances.
            Returns:
                list: List of aggregated semantic distance scores.
        """
        ## Compute the aggregated semantic distance score
        ### The score is computed as 1 - cosine distance
        return [1 - c for c in cosines]

    def aggregated_semantic_coherence_score(
        self,
        semantic_coherences
        ):
        """
            Compute the aggregated semantic coherence score based on semantic coherences.
            Args:
                semantic_coherences (list): List of semantic coherences.
            Returns:
                list: List of aggregated semantic coherence scores.
        """
        ## Compute the aggregated semantic coherence score
        ### The score is computed as 1 - semantic coherence
        return [1 - sc for sc in semantic_coherences]

    # @staticmethod
    def aggregated_global_semantic_score(
        self,
        cosines, 
        semantic_coherences
        ):
        """
            Compute the aggregated global semantic score based on cosine distances and semantic coherences.
            Args:
                cosines (list): List of cosine distances.
                semantic_coherences (list): List of semantic coherences.
            Returns:
                list: List of aggregated global semantic scores.
        """
        ## Compute the aggregated global semantic score
        ### The score is computed as the average of the semantic distance and coherence scores
        sem_dist = self.aggregated_semantic_distance_score(cosines)
        sem_coher = self.aggregated_semantic_coherence_score(semantic_coherences)
        return [(sd + sc)/2 for sd, sc in zip(sem_dist, sem_coher)]

    # @staticmethod
    def aggregated_semantic_score(
        self,
        local_semantic_scores, 
        global_semantic_scores
        ):
        """
            Compute the aggregated semantic score based on local and global semantic scores.
            Args:
                local_semantic_scores (list): List of local semantic scores.
                global_semantic_scores (list): List of global semantic scores.
            Returns:
                list: List of aggregated semantic scores.
        """
        ## Compute the aggregated semantic score
        ### The score is computed as a weighted average of the local and global semantic scores
        ### The weights are 0.25 for local semantic scores and 0.75 for global semantic scores
        return [(1/4 * ls + 3/4 * gs) for ls, gs in zip(local_semantic_scores, global_semantic_scores)]

    # @staticmethod
    def compute_dataset_stats(
        self,
        df
        ):
        """
            Compute the statistics of the dataset.
            Args:
                df (pd.DataFrame): The dataset.
            Returns:
                dict: A dictionary containing the mean of each column in the dataset.
        """
        ## Compute the mean of each (numeric) column in the dataset
        return {
            metric: round(100*np.mean(df[metric]),2) for metric in df.columns if df[metric].dtype != 'O'
            }
