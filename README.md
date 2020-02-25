# Enhanced UD Parsing

Code for enhanced UD parsing for the IWPT/UD shared task.

Run code in its current format:
```
allennlp train -f configs/enhanced_parser.jsonnet -s logs/enhanced_parser/ --include-package tagging
```

At the moment, the parser doesn't handle sentences with ellipsis so the parser is parsing `data/UD_English-EWT/en_ewt-ud-train_no_ellipsis.conllu` which is basically the same training/dev set but with these sentences removed.

The `enhanced_parser` parser can compute labeled and unlabeled f1 and gets around 86% LF1 on `en_ewt-ud-dev_no_ellipsis.conllu`: https://tensorboard.dev/experiment/8UZQ5QHVRb2mTBGySyRmig/#scalars

`biaffine_parser_enhanced` is a multi-task parser, i.e. computes the basic tree as well as the enhanced graph and sums their respective losses but training is unstable: https://tensorboard.dev/experiment/wqLyYh4aTRKcKPJei9GtRQ/#scalars
It might need separate encoders e.g. shared and task-specific encoders or else some more debugging.
