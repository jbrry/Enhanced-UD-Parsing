{
  dataset_reader: {
    type: 'conll_03_reader',
    lazy: false
  },
  train_data_path: 'data/UD_English-EWT/en_ewt-ud-train.conllu',
  validation_data_path: 'data/UD_English-EWT/en_ewt-ud-dev.conllu',
  trainer: {
    cuda_device: -1
  }
}

