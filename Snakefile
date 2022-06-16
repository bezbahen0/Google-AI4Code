
rule train_xgbranker:
    input:
        "data/featurized/xgb_data.pkl"
    output:
        "data/models/xgbranker.ubj"
    shell:
        "python -m src.train --data {input} --output {output}"


rule featurize_xgb_data:
    input:
        "data/clean/train_all_cleaned.parquet"
    output:
        "data/featurized/xgb_data.pkl"
        "data/featurized/tfidf_vocabulary.pkl"
        "data/featurized/tfidf_idf.pkl"
    shell:
        "python -m src.featurize --data {input} --output {output[0]} --task xgbranker"


rule clean_data:
    input:
        "data/merged/train_all.parquet"
    output:
        "data/clean/train_all_cleaned.parquet"
    shell:
        "python -m src.clean --data {input} --output {output}"


rule join_data:
    input:
        "data/raw/train_orders.csv",
        "data/raw/train_ancestors.csv",
        "data/raw/train"
    output:
        "data/merged/train_all.parquet"
    shell:
        "python -m src.merging --train_orders {input[0]} --train_ancestors {input[1]} --train {input[2]} --output {output}"