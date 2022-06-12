

#rule all:
#    input:
#        "data/raw"
#

rule featurize_xgb_data:
    input:
        #"data/clean/train_all_cleaned.parquet"
        'data/processed/train_all.parquet'
    output:
        "data/featurized/xgb_data.pkl"
    shell:
        "python -m src.featurize --data {input} --output {output} --task xgbranker"


#rule clean_data:
#    input:
#        "data/processed/train_all.parquet"
#    output:
#        "data/clean/train_all_cleaned.parquet"
#    shell:
#        "python -m src.clean --data {input} --output {output}"


rule join_data:
    input:
        "data/raw/train_orders.csv",
        "data/raw/train_ancestors.csv",
        "data/raw/train"
    output:
        "data/processed/train_all.parquet"
    shell:
        "python -m src.processing --train_orders {input[0]} --train_ancestors {input[1]} --train {input[2]} --output {output}"