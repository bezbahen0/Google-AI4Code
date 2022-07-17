rule all:
    input:
#        "data/models/xgbranker.ubj",
        "data/models/distilbert-5000-new-featurization-128-max_length.bin"


#rule train_xgbranker:
#    input:
#        "data/featurized/xgb_data.pkl"
#    output:
#        "data/models/xgbranker.ubj"
#    shell:
#        "python -m src.train --data {input} --output {output} --task xgbranker"


rule train_transformer:
    input:
        "data/featurized/transformer_data.parquet",
        "data/featurized/transformer_features.json"
    output:
        "data/models/distilbert-5000-new-featurization-128-max_length.bin"
    shell:
        '''
        python -m src.train \
            --data {input[0]} \
            --output {output} \
            --task transformer \
            --features_data_path {input[1]} \
            --model_name_or_path 'distilbert-base-uncased' \
            --md_max_len 128 \
            --total_max_len 128 \
            --accumulation_steps 4 \
            --batch_size 24 \
            --n_workers 6 \
            --epochs 5\
        '''


rule featurize_transformer_data:
    input:
        "data/merged/train_all.parquet"
        #"data/translated/train_all_translated.parquet"
        #"data/clean/train_all_cleaned.parquet",
    output:
        "data/featurized/transformer_data.parquet",
        "data/featurized/transformers_data_all.parquet",
        "data/featurized/transformer_features.json"
    shell:
        '''
        python -m src.featurize \
            --data {input} \
            --output {output[0]} \
            --task transformer \
            --processed_out_path {output[1]} \
            --features_out_path {output[2]} \
            --num_selected_code_cells 20 \
            --mode train 
        '''


#rule featurize_xgb_data:
#    input:
#        #"data/clean/train_all_cleaned.parquet"
#        "data/translated/train_all_translated.parquet"
#    output:
#        "data/featurized/xgb_data.pkl",
#        "data/featurized/tfidf_voc.pkl",
#        "data/featurized/tfidf_idf.pkl"
#    shell:
#        '''
#        python -m src.featurize \
#            --data {input} \
#            --output {output[0]} \
#            --task xgbranker \
#            --tfidf_voc_path {output[1]} \
#            --tfidf_idf_path {output[2]} \
#            --mode train
#        '''


rule translate_markdowns_cells:
    input:
       "data/clean/train_all_cleaned.parquet",
       "data/pretrained_models/lid.176.bin",
       "data/pretrained_models/converted",
       "data/pretrained_models/transformers"
    output:
        "data/translated/train_all_translated.parquet",
        "data/translated/languages.csv"
    shell:
        '''
        python -m src.translation \
            --data {input[0]}  \
            --output {output[0]} \
            --lang_ident_out {output[1]} \
            --fasttext_ident_path {input[1]} \
            --target_lang en \
            --marianmt_models_dir_path {input[2]}\
            --tokenizers_dir_path {input[3]} \
            --batch_size 256
        '''


rule clean_data:
    input:
        "data/merged/train_all.parquet"
    output:
        "data/clean/train_all_cleaned.parquet"
    shell:
        "python -m src.clean --data {input} --output {output} --clear all"


rule merge_data:
    input:
        "data/raw/train_orders.csv",
        "data/raw/train_ancestors.csv",
        "data/raw/train"
    output:
        "data/merged/train_all.parquet"
    shell:
        '''
        python -m src.merging  \
            --train_orders {input[0]} \
            --train_ancestors {input[1]}  \
            --mode train \
            --data {input[2]} \
            --output {output} \
            --num_notebooks 5000 
        '''