# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.

from nltk.tokenize import word_tokenize
import language_tool_python
import networkx as nx
import numpy as np
import benepar
import spacy
import nltk


class MorphologicalAnalyzer:
    """
        MorphologicalAnalyzer class for analyzing morphological hallucinations in sentences.
        This class uses a grammar checker and a language model to parse sentences and detect grammatical errors.
    """

    def __init__(
        self, 
        grammar_tool_language='en-US', 
        language_model='en_core_web_sm'
        ):
        """
            Initialize the MorphologicalAnalyzer with a grammar checker and a language model.
            Args:
                grammar_tool_language (str): The language for the grammar checker.
                language_model (str): The language model for parsing sentences.
        """
        ## Initialize grammar checker and language model
        self.grammar_tool = language_tool_python.LanguageToolPublicAPI(grammar_tool_language)
        self.nlp = spacy.load(language_model)
        self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    def _parse_sentence(
        self, 
        text
        ):
        """
            Parse the sentence using spaCy and Benepar.
            Args:
                text (str): The text to parse.
            Returns:
                dict: A dictionary containing the parsed tokens, POS tags, dependency tags, and heads.
        """
        ## Parse the sentence 
        doc = self.nlp(text)
        ## Extract the dependency parse
        dependency_parse = {
            'tokens': [token.text for token in doc],
            'pos_tags': [token.pos_ for token in doc],
            'dep_tags': [token.dep_ for token in doc],
            'heads': [token.head.i for token in doc]
            }
        ## Build the dependency tree
        dependency_tree = []
        for token in doc:
            dependency_tree.append({
                'token': token.text,
                'pos': token.pos_,
                'dep': token.dep_,
                'head': token.head.text,
                'head_pos': token.head.pos_,
                'children': [child.text for child in token.children]
            })
        ## Extract the constituency parse
        try:
            sent = list(doc.sents)[0]
            constituency_parse = sent._.parse_string
        except:
            constituency_parse = "Constituency parsing not available"
        ## Return the parsed sentence
        return {
            'doc': doc,
            'dependency_parse': dependency_parse,
            'dependency_tree': dependency_tree,
            'constituency_parse': constituency_parse
            }

    def _compare_syntax_trees(
        self, 
        reference_parse, 
        hypothesis_parse
        ):
        """
            Compare the syntax trees of reference and hypothesis sentences.
            Args:
                reference_parse (dict): The parsed reference sentence.
                hypothesis_parse (dict): The parsed hypothesis sentence.
            Returns:
                float: The structural divergence score.
        """
        ## Extract the dependency trees
        ref_deps = set((item['head'], item['dep'], item['token']) for item in reference_parse['dependency_tree'])
        hyp_deps = set((item['head'], item['dep'], item['token']) for item in hypothesis_parse['dependency_tree'])
        ## Calculate Jaccard similarity (|A ∩ B| / |A ∪ B|)
        intersection = len(ref_deps.intersection(hyp_deps))
        union = len(ref_deps.union(hyp_deps))
        jaccard_similarity = intersection / union if union > 0 else 0
        ## Calculate structural divergence
        syntax_divergence = 1 - jaccard_similarity
        ## Make the syntax divergence score between 0 and 1
        syntax_divergence = max(0, min(1, syntax_divergence))
        ## Return the structural divergence score
        return syntax_divergence

    def _detect_grammatical_errors(self, text):
        """
            Detect grammatical errors in the text using a grammar checker.
            Args:
                text (str): The text to check for grammatical errors.
            Returns:
                dict: A dictionary containing the total number of errors and their categories.
        """
        ## Define the matching rules
        matches = self.grammar_tool.check(text)
        ## Define the error categories
        error_categories = {
            'spelling': 0,
            'grammar': 0,
            'punctuation': 0,
            }
        ## Count the errors
        for match in matches:
            if 'spell' in match.message.lower():
                error_categories['spelling'] += 1
            elif 'grammar' in match.message.lower():
                error_categories['grammar'] += 1
            elif 'punctuat' in match.message.lower():
                error_categories['punctuation'] += 1
        ## Return the total number of errors and their categories
        return {
            'total_errors': len(matches),
            'error_categories': error_categories
            }

    def morphological_hallucination_score(
        self, 
        reference, 
        hypothesis
        ):
        """
            Compute the morphological hallucination score for a given reference and hypothesis sentence.
            Args: 
                reference (str): The reference sentence.
                hypothesis (str): The hypothesis sentence.
            Returns:
                dict: A dictionary containing the structural divergence and grammatical errors.
        """
        ## Check if the reference and hypothesis are the same
        # If they are the same, return 0 for all measures
        if reference == hypothesis:
            return {
                'structural_divergence': 0.0,
                'grammatical_errors': {
                    'total_errors': 0,
                    'error_categories': {
                        'spelling': 0,
                        'grammar': 0,
                        'punctuation': 0
                        }
                    }
                }
        ## Parse the sentences
        reference_parse = self._parse_sentence(reference)
        hypothesis_parse = self._parse_sentence(hypothesis)
        ## Compare the syntax trees and compute the structural divergence
        structural_divergence = self._compare_syntax_trees(reference_parse, hypothesis_parse)
        ## Detect grammatical errors in the hypothesis
        grammatical_errors = self._detect_grammatical_errors(hypothesis)
        ## Return the scores
        return {
            'structural_divergence': structural_divergence,
            'grammatical_errors': grammatical_errors,
            }