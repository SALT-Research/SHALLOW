# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License CC-BY-SA 4.0
# You may not use this file except in compliance with the License.

from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import evaluate
import torch


class SemanticAnalyzer:
    """
        SemanticAnalyzer class for analyzing semantic coherence and similarity between sentences.
        This class uses a local semantic model, a global semantic model, and an NLI model to compute semantic scores.
    """ 

    def __init__(
        self, 
        local_model_name='bert-base-uncased', 
        global_model_name='nli-roberta-base-v2',
        nli_model_name='facebook/bart-large-mnli',
        device='cuda'
        ):
        """
            Initialize the SemanticAnalyzer with a local and global semantic model.
            Args:
                global_model_name (str): The name of the global semantic model.
                local_model_name (str): The name of the local semantic model.
                nli_model_name (str): The name of the NLI model.
                device (str): The device to use for computation (e.g., 'cuda' or 'cpu').
        """
        ## Set the device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')    
        ## Load local semantic model
        self.local_tokenizer = AutoTokenizer.from_pretrained(local_model_name)
        self.local_model = AutoModel.from_pretrained(local_model_name).to(self.device)
        ## Load global semantic model   
        self.global_model = SentenceTransformer(global_model_name).to(self.device)
        ## Load NLI model
        self.nli_model = pipeline("text-classification", model=nli_model_name, device=0)

    def _compute_semantic_similarity(
        self, 
        embeddings
        ):
        """
            Compute the semantic similarity between two embeddings.
            Args:
                embeddings (list): A list of two embeddings.
            Returns:
                float: The cosine similarity between the two embeddings.
        """
        ## Compute the cosine similarity between the two embeddings
        with torch.no_grad():
            cosine_sim = util.pytorch_cos_sim(embeddings[0].cpu(), embeddings[1].cpu())
        ## Return the cosine similarity as a float
        return cosine_sim.item()

    def _bertnli_semantic_score(
        self, 
        reference, 
        hypothesis
        ):
        """
            Compute the BERTScore and NLI-based contradiction penalty for a given reference and hypothesis sentence.
            Args:
                reference (str): The reference sentence.
                hypothesis (str): The hypothesis sentence.
            Returns:
                tuple: A tuple containing the final semantic score, BERTScore F1, NLI label, and entailment probability.
        """
        ## Load the BERTScore metric
        bertscore = evaluate.load("bertscore")
        ## Compute the BERTScore
        bs = bertscore.compute(predictions=[hypothesis], references=[reference], lang="en")
        bert_f1 = bs['f1'][0]
        ## Compute NLI-based contradiction penalty
        nli_input = f"{reference} </s> {hypothesis}"
        nli_result = self.nli_model(nli_input, truncation=True)[0]
        label = nli_result['label']
        score = nli_result['score']
        ## Compute the entailment probability based on the NLI label
        if label.lower() == "entailment":
            entailment_prob = score
        elif label.lower() == "neutral":
            entailment_prob = score * 0.5   ## partially helpful
        else:  ## CONTRADICTION
            entailment_prob = 0.0           ## completely penalize
        ## Combine for final semantic score
        final_score = bert_f1 * entailment_prob
        ## Return the final semantic score, BERTScore F1, NLI label, and entailment probability
        return round(final_score, 4), bert_f1, label, round(entailment_prob, 4)

    def _measure_semantic_coherence(
        self, 
        reference, 
        hypothesis
        ):
        """
            Measure the semantic coherence between a reference and hypothesis sentence.
            Args:
                reference (str): The reference sentence.
                hypothesis (str): The hypothesis sentence.
            Returns:
                float: The semantic coherence score.
        """
        ## Check if the reference and hypothesis are the same
        # If they are the same, return 1.0
        # If one of them is empty, return 0.0
        if reference == hypothesis:
            return 1.0
        elif reference.strip() == "" or hypothesis.strip() == "":
            return 0.0
        try:
            ## Compute the semantic coherence using a modified version of BERTScore
            score, raw_bert, label, entail_prob = self._bertnli_semantic_score(
                str(reference), 
                str(hypothesis)
                )
        except Exception as e:
            print(f"Error: {e}")
            score = 0.0
        ## Return the semantic coherence score
        return score

    def _get_local_context_embedding(
        self, 
        text
        ):
        """
            Compute the context embedding for a given text using the local model.
            Args:
                text (str): The text to compute the context embedding for.
            Returns:
                torch.Tensor: The context embedding for the given text.
        """
        ## Tokenize the text and convert it to tensors
        inputs = self.local_tokenizer(
            text, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True
            )
        ## Compute the context embedding using the local model
        with torch.no_grad():
            outputs = self.local_model(**inputs.to(self.device))
        ## Return the mean of the last hidden state as the context embedding
        return outputs.last_hidden_state.mean(dim=1)

    def _compute_local_semantic_coherence(
        self, 
        ref_window, 
        hyp_window
        ):
        """
            Compute the local semantic coherence between a reference window and a hypothesis window.
            Args:
                ref_window (list): The reference window.
                hyp_window (list): The hypothesis window.
            Returns:
                float: The local semantic coherence score.
        """
        ## Compute the context embeddings for the reference and hypothesis windows
        ref_embedding = self._get_local_context_embedding(' '.join(ref_window))
        hyp_embedding = self._get_local_context_embedding(' '.join(hyp_window))
        ## Compute the cosine similarity between the reference and hypothesis embeddings
        ref_vec = ref_embedding.detach().cpu().numpy().flatten()
        hyp_vec = hyp_embedding.detach().cpu().numpy().flatten()
        cosine_similarity = np.dot(ref_vec, hyp_vec) / (np.linalg.norm(ref_vec) * np.linalg.norm(hyp_vec))
        ## Return the cosine similarity as the local semantic coherence score
        return cosine_similarity

    def local_semantic_score(
        self, 
        reference, 
        hypothesis, 
        window_sizes=[1, 2, 3]
        ):
        """
            Compute the local semantic score for a given reference and hypothesis sentence.
            Args: 
                reference (str): The reference sentence.
                hypothesis (str): The hypothesis sentence.
                window_sizes (list): List of window sizes to compute local semantic coherence.
            Returns:
                dict: A dictionary containing the local semantic scores for each window size.
        """
        ## Check if the reference and hypothesis are the same
        # If they are the same, return 1 for all window sizes
        # If one of them is empty, return 0 for all window sizes
        if reference == hypothesis:
            return {f'window_size_{w}': 1.0 for w in window_sizes}
        elif reference.strip() == "" or hypothesis.strip() == "":
            return {f'window_size_{w}': 0.0 for w in window_sizes}
        ## Convert the reference and hypothesis to lowercase and split into words
        r_words = reference.lower().split()
        h_words = hypothesis.lower().split()
        ## Compute the local semantic coherence for each window size
        local_semantic_scores = {}
        for window_size in window_sizes:
            lcs_window = {}
            for i in range(len(h_words) - window_size + 1):
                h_window = h_words[i:i + window_size]
                scores = []
                for j in range(len(r_words) - window_size + 1):
                    r_window = r_words[j:j + window_size]
                    score = self._compute_local_semantic_coherence(r_window, h_window)
                    scores.append(score)
                lcs_window[i] = max(scores) if scores else 0
            ## Compute the average local semantic coherence for the current window size
            local_semantic_scores[f'window_size_{window_size}'] = sum(lcs_window.values()) / max(len(r_words), len(h_words))
        ## Return the local semantic scores
        return local_semantic_scores


    def global_semantic_score(
        self, 
        reference, 
        hypothesis
        ):
        """
            Compute the global semantic score for a given reference and hypothesis sentence.
            Args: 
                reference (str): The reference sentence.
                hypothesis (str): The hypothesis sentence.
            Returns:
                dict: A dictionary containing the global semantic scores.
        """
        ## Check if the reference and hypothesis are the same
        # If they are the same, return 1 for all measures
        # If one of them is empty, return 0 for all measures
        if reference == hypothesis:
            return {
                'global_semantic_cosine_similarity': 1.0,
                'global_semantic_coherence': 1.0
                }
        elif reference.strip() == "" or hypothesis.strip() == "":
            return {
                'global_semantic_cosine_similarity': 0.0,
                'global_semantic_coherence': 0.0
                }
        ## Encode the reference and hypothesis using the semantic model
        with torch.no_grad():
            embeddings = self.global_model.encode(
                [reference, hypothesis], 
                convert_to_tensor=True
                )
        ## Compute the global semantic cosine similarity
        global_sem_cos_sim = self._compute_semantic_similarity(embeddings)
        ## Compute the global semantic coherence
        semantic_coherence = self._measure_semantic_coherence(reference, hypothesis)
        ## Return the global semantic scores
        return {
            'global_semantic_cosine_similarity': global_sem_cos_sim,
            'global_semantic_coherence': semantic_coherence,
            }
