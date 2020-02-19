from typing import Dict, Any, List, Tuple
from overrides import overrides
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
#from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

sentence_index = 0

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
        Runs the underlying model, and adds the ``"words"`` to the output, also tags and parse??.
        """
        sentence = json_dict["sentence"]

        tokens = sentence.split()  # Simple tokenization, can do better though!
        tokens = str(tokens)

        
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        if "@@UNKNOWN@@" not in self._model.vocab._token_to_index["arc_tags"]:
            print("found unknown label")
            # Handle cases where the labels are present in the test set but not training set
            # https://github.com/Hyperparticle/udify/blob/b6a1173e7e5fc1e4c63f4a7cf1563b469268a3b8/udify/predictors/predictor.py
            self._predict_unknown(instance)

        outputs = self._model.forward_on_instance(instance)
        #print(outputs)
        return sanitize(outputs)

    def _predict_unknown(self, instance: Instance):
        """
        Maps each unknown label in each namespace to a default token
        :param instance: the instance containing a list of labels for each namespace
        from: https://github.com/Hyperparticle/udify/blob/b6a1173e7e5fc1e4c63f4a7cf1563b469268a3b8/udify/predictors/predictor.py
        """
        def replace_tokens(instance: Instance, namespace: str, token: str):
            if namespace not in instance.fields:
                return

            instance.fields[namespace].labels = [label
                                                 if label in self._model.vocab._token_to_index[namespace]
                                                 else token
                                                 for label in instance.fields[namespace].labels]

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        global sentence_index

        sentence_index += 1
        sent_id = ('# sent_id = ' + str(sentence_index))
        text = ('# text = ' + ' '.join(w for w in outputs["tokens"]))

        word_count = len([word for word in outputs["tokens"]])
        lines = zip(*[outputs[k] if k in outputs else ["_"] * word_count
                      for k in ["tokens", "lemmas", "pos", "xpos", "feats",
                                "arcs", "arc_tags"]])

        output_lines = []
        for i, line in enumerate(lines):
            line = [str(l) for l in line]

            row = "\t".join(line) + "".join(["\t_"] * 2)
            output_lines.append(row)

        output_lines = [sent_id] + [text] + output_lines
        return "\n".join(output_lines) + "\n\n"