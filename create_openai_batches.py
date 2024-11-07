import re
import json
import tiktoken
from tqdm import tqdm
from pathlib import Path


tokenizer = tiktoken.encoding_for_model('gpt-4o')

all_docs_fp = Path.home() / r"Box\Fed-Register\all_doc_info.json"
ALL_DATA = json.load(open(all_docs_fp, "r"))
INPUT_DIR = Path.home() / r"Box\Fed-Register\Final-Rule-txts"
OUT_DIR = Path.home() / (r"Box\Fed-Register\Final-Rule-Batches-20241102")
OUT_DIR.mkdir(exist_ok=True)

PROMPT = """Extract academic references from the text below and return them using the json format below. 
The json object should have the following keys: "citation", "title", "authors", "year", "journal", "publisher", "location", "volume", "pages", "doi", and "url". Where the citation key should contain the full citation of the paper.
If there is not enough information to completely fill out the json, return as much as possible. If the author is "et. al." or not a person (e.g. "EPA"), flag the citation with "et_al_flag" or "non_person_author_flag" respectively.
If available, provide the first and last name of each author. If there are multiple authors, format the authors as a python list of strings.
If there are no references in the text, DO NOT return anything. 

Json format to be precisely followed:
```json
{
    "citation": "Author1, Author2, (Year). Title of the paper. Location: Journal Publisher, Volume, Pages. DOI. URL"
    "title": "Title of the paper",
    "authors": ["FirstName Lastname 1","FirstName LastName 2"],
    "year": "Year",
    "journal": "Journal",
    "publisher": "Publisher",
    "location": "Location",
    "volume": "Volume",
    "pages": "Pages",
    "doi": "DOI",
    "url": "URL"
    "et_al_flag": "True/False"
    "non_person_author_flag": "True/False"
}
```
"""


def extract_citations(text):
    legal_regex = re.compile(r"\b\d{1,2}\sC?FR\s\d{1,5}(\([a-z]\d*\))*\s*(\(.+?\))?|\b\d{1,2}\sU\.S\.C\.\s\d{1,5}(\([a-zA-Z]\d*\))*|\b\d{1,2}\sU\.S\.C\.\s\d{1,5}(\([a-zA-Z]\d*\))*(\s*,\s*\d{1,2}\sU\.S\.C\.\s\d{1,5}(\([a-zA-Z]\d*\))*)*(\s*and\s*\d{1,2}\sU\.S\.C\.\s\d{1,5}(\([a-zA-Z]\d*\))*)?")
    sections = re.split(r"-{10,}", text)
    citations = ""
    for section in sections:
        section = section.strip()

        # don't want blank or content sections
        if (not section) or not (re.match(r"^\\\d+\\", section)):
            continue

        # Find all citations in section
        pattern = re.compile(r"\\(\d+)\\\s+(.*?)(?=(?:\\\d+\\|\Z))", re.DOTALL)
        matches = pattern.findall(section)

        # Store citation numbers and their texts
        for match in matches:
            citation_text = match[1].replace("\n", " ").strip()
            citations += citation_text + "\n"
            
    citations = citations.splitlines()
    citations = [line.strip() for line in citations if line.strip()]
    citations = [line for line in citations if not (line.strip().lower().startswith("ibid."))]
    citations = [line for line in citations if not (line.strip().lower().startswith("id."))]
    citations = [line for line in citations if not (line.strip().lower().startswith("see sec."))]
    citations = [line for line in citations if not (line.strip().lower().startswith("sec."))]
    citations = [line for line in citations if not (line.strip().lower().startswith("see supra"))]
    citations = [line for line in citations if not (line.strip().lower().startswith("supra"))]
    citations = [line for line in citations if not (line.strip().startswith("ISO"))]
    citations = [line for line in citations if not (legal_regex.search(line.strip()))]
    citations = "\n".join(citations)
    return citations


def chunk_text(prompt, text, max_tokens=2500):
    lines = []
    total_tokens = 0 
    for line in text.splitlines():
        encoded_line = tokenizer.encode(line)
        lines.append(encoded_line)
        total_tokens += len(encoded_line)

    chunk_size = max_tokens - len(tokenizer.encode(prompt))

    chunks = []
    current_chunk = []
    current_chunk_length = 0
    for line in lines:
        line_length = len(line)
        if current_chunk_length + line_length <= chunk_size:
            current_chunk.append(line)
            current_chunk_length += line_length
        else:
            chunks.append(current_chunk)
            current_chunk = [line]
            current_chunk_length = line_length
    if current_chunk:
        chunks.append(current_chunk)

    decoded_chunks = []
    for chunk in chunks:
        decoded_lines = [tokenizer.decode(line) for line in chunk]
        decoded_chunk = "\n".join(decoded_lines)
        decoded_chunks.append(decoded_chunk)

    chunks = decoded_chunks
    return chunks, total_tokens


def create_batch_file(prompt, text, id, year):
    content = {"custom_id": id, 
               "method": "POST",
               "url": "/v1/chat/completions",
               "body": {
               "model": "gpt-4o",
               "messages": [
                   {"role": "system", "content": "You are an assistant that extracts academic references from text."},
                   {"role": "user","content": prompt},
                   {"role": "user", "content": text}
                   ],
                "n": 1,
                "temperature":0.5}
    }
    fp = OUT_DIR / f"batch_file_{year}.jsonl"
    with open(fp, "a") as f:
        json.dump(content, f)
        f.write("\n")
        
    return fp


def main():
    file_paths = {}
    for year in tqdm(range(1990, 2025), desc="Finding files"):
        doc_ids = [item['document_number'] for item in ALL_DATA if int(item['publication_date'][:4]) == year]
        file_paths[year] = set([INPUT_DIR / f"{doc_id}.txt" for doc_id in doc_ids if (INPUT_DIR / f"{doc_id}.txt").exists()])

    
    yearly_batch_files = set()
    for year, files in file_paths.items():
        for file in tqdm(files, desc=f"Processing files for {year}"):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
            refs = extract_citations(text)
            if not refs:
                continue
            chunks, tok_count = chunk_text(PROMPT, refs)
            
            i = 1 # To differentiate different chunks from the same document
            for chunk in chunks:
                yearly_batch_files.add(create_batch_file(PROMPT, chunk, f"{file.stem}_{i}", year))
                i += 1
                
    for file in yearly_batch_files:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        max_lines = 25
        file_count = 1
        for i in range(0, len(lines), max_lines):
            new_file_path = file.with_name(f"{file.stem}_part{file_count}{file.suffix}")
            with open(new_file_path, "w", encoding="utf-8") as new_file:
                new_file.writelines(lines[i:i + max_lines])
                file_count += 1
                
        file.unlink()

if __name__ == "__main__":
    main()