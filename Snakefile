config_path = "configs/transformer.yml"


configfile: config_path


# Trained model name
model_description = f"{config['processing']}"
model_description += f""

model_name = config["model_name"] + "_" + model_description

# Submit name
model_featured = ""
submit_description = model_description + model_featured
submit_name = config["model_name"] + "_" + submit_description

processing = config["processing"]


rule all:
    input: 
        f"data/models/{model_name}.bin"


rule train_transformer:
    input:
        "data/featurized/transformer_data_id.parquet",
        "data/featurized/transformer_features.json",
    output:
        f"data/models/{model_name}.bin"
    shell:
        "python -m src.train "
        "    --data {input[0]} "
        "    --output {output} "
        "    --features_data_path {input[1]} "
        f"    --config {config_path} "
        


rule featurize_transformer_data:
    input:
        f"data/merged/train_{config['num_notebooks']}.parquet"
    output:
        "data/featurized/transformer_data_id.parquet",
        "data/featurized/transformers_data_source.parquet",
        "data/featurized/transformer_features.json"
    shell:
        "python -m src.featurize "
        "    --data {input} "
        "    --output {output[0]} "
        "    --processed_out_path {output[1]} "
        "    --features_out_path {output[2]} "
        "    --mode train "
        f"   --config {config_path} "
        

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
        f"data/merged/train_{config['num_notebooks']}.parquet"
    shell:
        "python -m src.merging   "
        "    --train_orders {input[0]}  "
        "    --train_ancestors {input[1]} "
        "    --mode train          "
        "    --data {input[2]}     "
        "    --output {output}     "
        f"    --config {config_path} "
