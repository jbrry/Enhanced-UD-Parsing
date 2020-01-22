{
  "random_seed": 711,
  dataset_reader: {
    type: 'universal_dependencies_enhanced',
    lazy: false,
    print_data: false
  },
  train_data_path: 'data/UD_English-EWT/en_ewt-ud-train.conllu',
  validation_data_path: 'data/UD_English-EWT/en_ewt-ud-dev.conllu',
  test_data_path: 'data/UD_English-EWT/en_ewt-ud-test.conllu',
  model: {
    type: 'dummy_lstm'
  },
  trainer: {
    cuda_device: -1
  }
}

