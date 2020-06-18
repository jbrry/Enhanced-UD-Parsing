# source: https://github.com/Hyperparticle/udify/blob/master/udify/predictors/predictor.py
# modified by James Barry, Dublin City University
# License: MIT License

"""
based on the implementation in: https://github.com/Hyperparticle/udify/blob/master/udify/predictors/predictor.py
"""

from typing import Dict, Any, List, Tuple
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

@Predictor.register("enhanced-predictor")
class EnhancedPredictor(Predictor):
    """
    Predictor that takes in a sentence and returns
    a set of heads and tags for it.
    Predictor for the :class:`~allennlp.models.BiaffineDependencyParser` model
    but extended to write conllu lines.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
    
    def predict(self, sentence: str) -> JsonDict: 
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"tokens"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = sentence.split()
        tokens = str(tokens)
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        conllu_metadata = outputs["conllu_metadata"]
        word_count = len([word for word in outputs["tokens"]])

        predicted_arcs = outputs["arcs"]
        predicted_arc_tags = outputs["arc_tags"]
        
        # changes None to "_"
        cleaned_heads = []
        predicted_heads = outputs["head_indices"]
        for head in predicted_heads:
            if type(head) != int:
                head = "_"
                cleaned_heads.append(head)
            else:
                cleaned_heads.append(head)
        outputs["head_indices"] = cleaned_heads
                     
        # dictionary mapping original conllu IDs (which contain float-values) to 1-indexed IDs as they appear in the sentence
        # if there are no elided tokens this is None
        original_to_new_indices = outputs["original_to_new_indices"]

        # create dict to store mappings from CoNLLU ids to dependency relations
        id_to_deprel_mappings = {conllu_id: [] for conllu_id in outputs["ids"]}

        for label_index, (head, dep) in enumerate(predicted_arcs):
            # change the head and dep to the original indices
            if type(original_to_new_indices) == dict:
                original_to_new_index_mappings = original_to_new_indices.items()
                for mapping in original_to_new_index_mappings:
                    # if head or dep is the newer index[1]; change back to the original index[0]
                    if head == mapping[1]:
                        head = mapping[0]
                    if dep == mapping[1]:
                        dep = mapping[0]
            
            if dep in id_to_deprel_mappings:
                id_to_deprel_mappings[dep].append((head, predicted_arc_tags[label_index]))

        # another dict to store the formatted deprels
        id_to_formatted_deprel_mappings = {}
        
        for conllu_id, pred_output in id_to_deprel_mappings.items():            
            # keep a list for words with multiple heads
            current_targets = []
            num_deprels = len(pred_output)
            for head_rel_tuple in pred_output:    
                target = ":".join(str(x) for x in head_rel_tuple)                               
                if num_deprels == 1:
                    id_to_formatted_deprel_mappings[conllu_id] = target                
                elif num_deprels > 1:
                    # add deprels to list so they can be joined
                    current_targets.append(target)            

            if num_deprels > 1:
                # pipe-join multiple deprels
                formatted_target = "|".join(str(x) for x in current_targets)
                id_to_formatted_deprel_mappings[conllu_id] = formatted_target

        #print("id_to_formatted_deprel_mappings", id_to_formatted_deprel_mappings)
           
        # restructure the outputs to match the CoNLLU format
        outputs["arc_tags"] = id_to_formatted_deprel_mappings.values()
       
        lines = zip(*[outputs[k] if k in outputs else ["_"] * word_count
                      for k in ["ids", "tokens", "lemmas", "upos", "xpos", "feats",
                                "head_indices", "head_tags", "arc_tags", "misc"]])

        multiword_map = None
        if outputs["multiword_ids"]:
            multiword_ids = [[id] + [int(x) for x in id.split("-")] for id in outputs["multiword_ids"]]
            multiword_forms = outputs["multiword_forms"]
            multiword_map = {start: (id_, form) for (id_, start, end), form in zip(multiword_ids, multiword_forms)}


        output_lines = []
        for i, line in enumerate(lines):
            line = [str(l) for l in line]

            # Handle multiword tokens
            if multiword_map and i+1 in multiword_map:
                id_, form = multiword_map[i+1]
                row = f"{id_}\t{form}" + "".join(["\t_"] * 8)
                output_lines.append(row)

            row = "\t".join(line)
            output_lines.append(row)

        output_lines = conllu_metadata + output_lines
        return "\n".join(output_lines) + "\n\n"
