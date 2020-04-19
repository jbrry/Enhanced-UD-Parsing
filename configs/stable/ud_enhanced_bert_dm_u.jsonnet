local bert_embedding_dim = 768;
local char_embedding_dim = 64;
local pos_embedding_dim = 100;
local embedding_dim = bert_embedding_dim + pos_embedding_dim + char_embedding_dim + char_embedding_dim;
local hidden_dim = 600;
local num_epochs = 50;
local patience = 10;
local batch_size = 16;
local learning_rate = 0.001;
local dropout = 0.33;
local input_dropout = 0.33;
local recurrent_dropout_probability = 0.33;

{
    "dataset_reader":{
        "type":"universal_dependencies_enhanced",
        "token_indexers": {
        "bert": {
          "type": "bert-pretrained",
          "pretrained_model": std.extVar("BERT_VOCAB"),
          "do_lowercase": false,
          "use_starting_offsets": true,
	  "truncate_long_sequences": false
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
      "type": "enhanced_parser",
      "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets"],
        "token_characters": ["token_characters"],
      },
      "token_embedders": {
        "bert": {
          "type": "bert-pretrained",
          "pretrained_model": std.extVar("BERT_WEIGHTS")
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
               "num_layers": 2,
               "bidirectional": true
               }
           }
        },
      },
      "pos_tag_embedding":{
        "embedding_dim": pos_embedding_dim,
        "vocab_namespace": "pos"
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
      "batch_size" : batch_size
    },
    "trainer": {
      "num_epochs": num_epochs,
      "grad_norm": 5.0,
      "patience": 12,
      "cuda_device": 0,
      "validation_metric": "+labeled_f1",
      "num_serialized_models_to_keep": 1,
      "optimizer": {
        "type": "dense_sparse_adam",
        "betas": [0.9, 0.9]
      }
    }
  }
