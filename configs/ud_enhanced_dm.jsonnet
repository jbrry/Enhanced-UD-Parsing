local word_embedding_dim = 100;
local char_embedding_dim = 64;
local tag_embedding_dim = 50;
local tag_combined_dim = 150;
local embedding_dim = word_embedding_dim + tag_combined_dim + char_embedding_dim + char_embedding_dim;
local hidden_dim = 400;
local patience = 10;
local learning_rate = 0.001;

{
  "random_seed":  std.parseInt(std.extVar("RANDOM_SEED")),
  "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
  "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
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
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("DEV_DATA_PATH"),
    //"test_data_path": std.extVar("TEST_DATA_PATH"),
    "model": {
      "type": "enhanced_dm_parser",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": word_embedding_dim,
            "sparse": true
           },
           "token_characters": {
             "type": "character_encoding",
             "embedding": {
               "embedding_dim": char_embedding_dim
             },
             "encoder": {
               "type": "lstm",
               "input_size": char_embedding_dim,
               "hidden_size": char_embedding_dim,
               "num_layers": 2,
               "bidirectional": true
             }
           }
        },
      },
      "lemma_tag_embedding":{
        "embedding_dim": tag_embedding_dim,
        "vocab_namespace": "lemmas",
        "sparse": true
      },
      "upos_tag_embedding":{
        "embedding_dim": tag_embedding_dim,
        "vocab_namespace": "upos",
        "sparse": true
      },
      "xpos_tag_embedding":{
        "embedding_dim": tag_embedding_dim,
        "vocab_namespace": "xpos",
        "sparse": true
      },
      "encoder": {
        "type": "stacked_bidirectional_lstm",
        "input_size": embedding_dim,
        "hidden_size": hidden_dim,
        "num_layers": 3,
        "recurrent_dropout_probability": 0.33,
        "use_highway": true
      },
      "arc_representation_dim": 500,
      "tag_representation_dim": 100,
      "dropout": 0.33,
      "input_dropout": 0.33,
    },
    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["tokens"],
        "batch_size": std.parseInt(std.extVar("BATCH_SIZE"))
      }
    },
    "evaluate_on_test": false,
    "trainer": {
      "num_epochs": std.parseInt(std.extVar("NUM_EPOCHS")),
      "grad_norm": 5.0,
      "patience": 10,
      "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
      "validation_metric": "+labeled_f1",
      "num_gradient_accumulation_steps": std.parseInt(std.extVar("GRAD_ACCUM_BATCH_SIZE")),
      "optimizer": {
        "type": "dense_sparse_adam",
        "betas": [0.9, 0.9]
      }
    }
  }
