# Enhanced UD Parsing

Code for enhanced UD parsing for the IWPT/UD shared task.

### Modifications
- parsing the `deps` column instead of `dep-rels` and `heads`.
- parse `deps` into correct format
- create a SequenceMultiLabelField to store data in list-of-lists format, e.g. for sequence labelling with multiple labels.
- create a `nested_sequence_cross_entropy` function to compute loss over multiple labels.

Run code in its current format:
```
allennlp train -f configs/enhanced_parser.jsonnet -s logs/enhanced_parser/ --include-package tagging
```
