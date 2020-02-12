local word_embedding_dim = 50;
local char_embedding_dim = 32;
local embedding_dim = word_embedding_dim + char_embedding_dim + char_embedding_dim;
local hidden_dim = 100;
local num_epochs = 50;
local patience = 10;
local batch_size = 4;
local learning_rate = 0.001;
local cuda_device = -1;

{
  "dataset_reader":{
    "type":"universal_dependencies_enhanced",
      "token_indexers": {
        "tokens": { 
        "type": "single_id" 
        },
        "token_characters": { 
        "type": "characters",
        "min_padding_length": 3
        }
      }
    },
  train_data_path: 'data/UD_English-EWT/train_sample.conllu',
  validation_data_path: 'data/UD_English-EWT/test_sample.conllu',
//  test_data_path: 'data/UD_English-EWT/en_ewt-ud-test.conllu',
  "model": {
      "type": "pos_tagger",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": word_embedding_dim
           },
           "token_characters": {
             "type": "character_encoding",
             "embedding": {
               "embedding_dim": char_embedding_dim,
             },
             "encoder": {
               "type": "lstm",
               "input_size": char_embedding_dim,
               "hidden_size": char_embedding_dim,
               "num_layers": 1,
               "bidirectional": true
             }
           }
        },
      },
      "encoder": {
        "type": "stacked_bidirectional_lstm",
        "input_size": embedding_dim,
        "hidden_size": hidden_dim,
        "num_layers": 2,
        "recurrent_dropout_probability": 0.33,
        "use_highway": true
      },
      "dropout": 0.33,
      "input_dropout": 0.33,
    },
    "iterator": {
      "type": "bucket",
      "sorting_keys": [["words", "num_tokens"]],
      "batch_size" : batch_size
    },
    "evaluate_on_test": true,
    "trainer": {
      "num_epochs": num_epochs,
      "grad_norm": 5.0,
      "patience": 50,
      "cuda_device": cuda_device,
      "validation_metric": "+accuracy",
      "num_serialized_models_to_keep": 3,
      "optimizer": {
        "type": "dense_sparse_adam",
        "betas": [0.9, 0.999]
      }
    }
  }
