{
    "dataset_reader":{
        //"type":"universal_dependencies"
        "type":"universal_dependencies_enhanced"
    },
  
  "train_data_path": "data/UD_English-EWT/en_ewt-ud-train_no_ellipsis.conllu",
  "validation_data_path": "data/UD_English-EWT/en_ewt-ud-dev_no_ellipsis.conllu",
  //"train_data_path": "data/UD_English-EWT/en_ewt-ud-train-ellided_only.conllu",
  //"train_data_path": "data/UD_English-EWT/en_ewt-ud-train.conllu",
  //"validation_data_path": "data/UD_English-EWT/en_ewt-ud-dev.conllu",
  //"validation_data_path": "data/UD_English-EWT/en_ewt-ud-train-ellided_only.conllu",
    "model": {
      "type": "enhanced_parser",
      //"type": "biaffine_parser_original",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": 100
          }
        }
      },
      "pos_tag_embedding":{
        "embedding_dim": 100,
        "vocab_namespace": "pos_tags"
      },
      "encoder": {
        "type": "stacked_bidirectional_lstm",
        "input_size": 200,
        "hidden_size": 600,
        "num_layers": 3,
        "recurrent_dropout_probability": 0.33,
        "use_highway": true
      },
      "arc_representation_dim": 500,
      "tag_representation_dim": 100,
      "dropout": 0.33,
      "input_dropout": 0.33,
      "initializer": [
        [".*feedforward.*weight", {"type": "xavier_uniform"}],
        [".*feedforward.*bias", {"type": "zero"}],
        [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
        [".*tag_bilinear.*bias", {"type": "zero"}],
        [".*weight_ih.*", {"type": "xavier_uniform"}],
        [".*weight_hh.*", {"type": "orthogonal"}],
        [".*bias_ih.*", {"type": "zero"}],
        [".*bias_hh.*", {"type": "lstm_hidden_bias"}]]
    },

    "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "num_tokens"]],
      "batch_size" : 32
    },
    "trainer": {
      "num_epochs": 10,
      "grad_norm": 5.0,
      "patience": 50,
      "cuda_device": 0,
      "validation_metric": "+labeled_f1",
      "optimizer": {
        "type": "dense_sparse_adam",
        "betas": [0.9, 0.9]
      }
    }
  }
