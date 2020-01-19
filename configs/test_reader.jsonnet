{
  dataset_reader: {
    type: 'universal_dependencies_enhanced',
    lazy: false,
    print_data: false
  },
  train_data_path: 'data/UD_English-EWT/en_ewt-ud-train_sample.conllu',
  //validation_data_path: 'data/UD_English-EWT/en_ewt-ud-dev.conllu',
  trainer: {
    cuda_device: -1
  }
}

