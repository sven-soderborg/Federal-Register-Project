import os
import time
import json
import shutil
import dotenv
import logging
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI, BadRequestError, RateLimitError

logging.basicConfig(filename=Path.home() / 'box/fed-register/logs/batching.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

dotenv.load_dotenv("G:/my drive/api_keys.env")
key = os.getenv("ADAMOPENAI")
open_client = OpenAI(api_key=key)


INPUT_DIR = Path.home() / (r"Box\Fed-Register\Final-Rule-Batches-20241102")
OUT_DIR = Path.home() / (r"Box\Fed-Register\Final-Rule-Batch-Results-20241102")
(OUT_DIR / "completed-batches").mkdir(exist_ok=True, parents=True)



def create_openai_batch(in_batch_file, year):
    input_file = open_client.files.create(file=open(in_batch_file, "rb"), purpose="batch")
    input_file_id = input_file.id
    batch = open_client.batches.create(input_file_id=input_file_id, 
                                       endpoint="/v1/chat/completions",
                                       completion_window="24h",
                                       metadata={
                                           "description": f"{year} Final-Rules"
                                        }
                                       )
    return batch



def process_completed_batch(batch):
    # Code to process the completed batch
    logging.info(f"Processing completed batch {batch.id}")
    result = open_client.files.content(open_client.batches.retrieve(batch.id).output_file_id).content
    name = batch.metadata["description"]
    with open(OUT_DIR / f"completed-batches/{name}.jsonl", "wb") as f:
        f.write(result)



def main():
    files = {f"{int(file.stem.split('_')[-2])}_{file.stem.split('_')[-1]}": file for file in INPUT_DIR.glob("*.jsonl")}
    files = dict(sorted(files.items(), key=lambda item: item[0], reverse=True))
    
    for year, file in tqdm(files.items(), desc="Creating OpenAI batches)"):
        if (OUT_DIR / f"completed-batches/{file.stem} Final-Rules.jsonl").exists():
            continue
        b = create_openai_batch(file, file.stem)
        worked = True
        while open_client.batches.retrieve(b.id).status != "completed":
            if open_client.batches.retrieve(b.id).status == "failed":
                logging.error(f"Batch {b.id} failed")
                worked = False
                break
            time.sleep(30)
        if worked:
            process_completed_batch(b)
main()