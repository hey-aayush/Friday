import logging
from pathlib import Path
import os
import hydra
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
import sentence_transformers.models as st_models
from sentence_transformers.losses import CosineSimilarityLoss
import transformers
from weblinx.utils.hydra import save_path_to_hydra_logs
import weblinx as wl
import shutil
import subprocess
import json
import os
from huggingface_hub import snapshot_download
from .processing import build_records_for_single_demo, build_formatters

def delete_demos(demo_dir):
  if os.path.isdir(demo_dir):
    shutil.rmtree(demo_dir)

def download_demo(demo_names,sample_data_dir):
  patterns = [f"demonstrations/{name}/*" for name in demo_names]
  ignore_patterns = [f"demonstrations/{name}/screenshots/*" for name in demo_names]
  ignore_patterns += [f"demonstrations/{name}/*.mp4" for name in demo_names]
  snapshot_download(
      "McGill-NLP/WebLINX-full", repo_type="dataset", local_dir=sample_data_dir, allow_patterns=patterns,ignore_patterns=ignore_patterns
  )

def return_split_data(split_path):
  with open(split_path, 'r') as file:
    data = json.load(file)
    return data

def fetch_next_demos(split_path,train_split,buffer_split,data,start, end):
  demo_names=data[train_split][start:end]
  data[buffer_split] = []
  data[buffer_split] = demo_names
  # Save the modified JSON back to the file
  with open(split_path, 'w') as file:
    json.dump(data, file, indent=2)
  return demo_names

def infer_optimizer(name):
    name = name.lower()

    if name == "adamw":
        return torch.optim.AdamW
    elif name == "adam":
        return torch.optim.Adam
    elif name == "adafactor":
        return transformers.Adafactor
    elif name == "sgd":
        return torch.optim.SGD
    else:
        raise ValueError(f"Unknown optimizer name: {name}")

def load_new_data(split_path,split,demo_dir,cfg,batch_size):
    demo_names = wl.utils.load_demo_names_in_split(split_path, split=split)
    demos = [wl.Demonstration(demo_name, demo_dir) for demo_name in demo_names]

    if cfg.project_name.endswith("testing"):
        demos = demos[:10]

    format_intent_input, _ = build_formatters()
    input_records = []
    logging.info(f"Number of demos: {len(demos)}. Starting building records.")
    for demo in tqdm(demos, desc="Building input records"):
        input_records.extend(
            build_records_for_single_demo(
                demo=demo,
                format_intent_input=format_intent_input,
                max_neg_per_turn=cfg.train.max_neg_per_turn,
                random_state=cfg.seed,
                # For training, we only want to include elements with valid uids
                # otherwise, we will be training on a lot of negative examples
                only_allow_valid_uid=True,
            )
        )

    logging.info(f"Number of input records: {len(input_records)}")

    train_examples = [
        InputExample(texts=[r["query"], r["doc"]], label=float(r["label"]))
        for r in tqdm(
            input_records, desc="Converting records to sentence-transformers input"
        )
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    return train_dataloader

def train_dmr(num_epochs,use_amp,torch_dtype,model,train_dataloader,train_loss,optim,warmup_steps,model_save_dir,scheduler,learning_rate):
    logging.info(f"Starting training for {num_epochs} epochs.")
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch_dtype):
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            optimizer_class=infer_optimizer(optim),
            warmup_steps=warmup_steps,
            output_path=str(model_save_dir),
            weight_decay=0.0,
            scheduler=scheduler,
            optimizer_params={"lr": learning_rate},
        )
    logging.info("Training complete.")

def start_batch_dmr_training(split_path,split,cfg,batch_size,num_epochs,use_amp,torch_dtype,model,train_loss,optim,warmup_steps,model_save_dir,scheduler,learning_rate,demo_dir,sample_data_dir):
  train_split = 'train'
  buffer_split = 'sample_train'
  data = return_split_data(split_path)
  total=len(data[train_split])
  step=60
  for idx in range(0,total,step):
    print(f"Clearing last demos")
    delete_demos(demo_dir)
    start,end = idx, min(total-1,idx+step)
    print(f"Fetching demos {idx}-{end}")
    demo_names = fetch_next_demos(split_path,train_split,buffer_split,data,start, end)
    print(f"Downloading demos {idx}-{end}")
    download_demo(demo_names,sample_data_dir)
    print(f"Training DMR on demos {idx}-{end}")
    train_dataloader=load_new_data(split_path,split,demo_dir,cfg,batch_size)
    train_dmr(num_epochs,use_amp,torch_dtype,model,train_dataloader,train_loss,optim,warmup_steps,model_save_dir,scheduler,learning_rate)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    torch.manual_seed(cfg.seed)

    model_name = cfg.model.name
    use_bf16 = cfg.model.use_bf16
    max_seq_length = cfg.model.max_seq_length
    optim = cfg.train.optim
    split = cfg.train.split
    learning_rate = cfg.train.learning_rate
    warmup_steps = cfg.train.warmup_steps
    batch_size = cfg.train.batch_size_per_device
    num_epochs = cfg.train.num_epochs
    scheduler = cfg.train.scheduler
    demo_dir = cfg.train.demo_dir
    sample_data_dir = cfg.train.sample_data_dir
    split_path = split_path = Path(cfg.data.split_path).expanduser()
    model_save_dir = Path(cfg.model.save_dir).expanduser()

    if use_bf16:
        torch_dtype = torch.bfloat16
        use_amp = False
    else:
        torch_dtype = torch.float32
        use_amp = True

    # Model loading
    word_embedding_model = st_models.Transformer(
        model_name, max_seq_length=max_seq_length
    )
    if cfg.train.gradient_checkpointing and hasattr(
        word_embedding_model.auto_model, "gradient_checkpointing_enable"
    ):
        word_embedding_model.auto_model.gradient_checkpointing_enable()

    pooling_model = st_models.Pooling(
        word_embedding_model.get_word_embedding_dimension()
    )
    if os.path.exists(model_save_dir):
        logging.info(f"Creating New Model")
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        logging.info(f"Loading Trained Model")
        model = SentenceTransformer(str('/mnt/batch/tasks/shared/LS_root/mounts/clusters/ashandilya-fhl-weblinx/code/Users/ashandilya_debug/FINAL/weblinx/checkpoints/dmr/sentence-transformers/all-MiniLM-L6-v2'))
        
    train_loss = CosineSimilarityLoss(model=model)

    start_batch_dmr_training(split_path,split,cfg,batch_size,num_epochs,use_amp,torch_dtype,model,train_loss,optim,warmup_steps,model_save_dir,scheduler,learning_rate,demo_dir,sample_data_dir)

    save_path_to_hydra_logs(save_dir=model_save_dir)

    return model_save_dir


if __name__ == "__main__":
    main()
