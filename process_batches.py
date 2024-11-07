import re
import json
import pandas as pd
import numpy as np
from pathlib import Path


def extract_json_objects(content):
    json_objects = []

    # Preprocess content to fix formatting issues
    content = re.sub(r'"\s*"(?=[a-zA-Z]+)', '",\n"', content)

    # Variables to track the position of curly braces
    start_index = -1
    brace_count = 0

    for i, char in enumerate(content):
        if char == "{":
            if brace_count == 0:
                start_index = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_index != -1:
                json_str = content[start_index : i + 1]
                try:
                    json_obj = json.loads(json_str)
                    json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    error_position = e.pos
                    error_context = json_str[
                        max(0, error_position - 30) : min(
                            len(json_str), error_position + 30
                        )
                    ]
                    print(f"Skipping invalid JSON: {json_str}\nError: {e}\nError context: {error_context}")

    return json_objects


def calculate_cost(content):
    in_tokens = 0
    out_tokens = 0
    tot_tokens = 0
    for item in content:
        in_tokens += int(
            item.get("response").get("body").get("usage").get("prompt_tokens")
        )
        out_tokens += int(
            item.get("response").get("body").get("usage").get("completion_tokens")
        )
        tot_tokens += int(
            item.get("response").get("body").get("usage").get("total_tokens")
        )

    print(
        f"Input tokens: {in_tokens}\nOutput tokens: {out_tokens}\nTotal tokens: {tot_tokens}"
    )
    print(
        f"Input price: ${(in_tokens / 1000000) * 1.25}\nOutput Price: ${(out_tokens / 1000000) * 5}\nTotal Price: ${((in_tokens / 1000000) * 1.25) + ((out_tokens / 1000000) * 5)}"
    )


def get_responses(directory: Path):
    content = []
    for file in directory.glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as file:
            for line in file.readlines():
                content.append(json.loads(line))
    calculate_cost(content)

    responses = {}
    for item in content:
        response = item.get("response").get("body").get("choices")[0].get("message")
        if response:
            response = response.get("content")
            id = item.get("custom_id")
            responses[id] = response

    return responses


def process_responses(responses: dict):
    dfs = []
    for id, response in responses.items():
        json_objects = extract_json_objects(response)
        df = pd.DataFrame(json_objects)
        df["docid"] = id.split("_")[0]
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)
    # all_data = all_data.drop(columns=["isbn", "issue", "article number"])

    col_invalid_values = {
        "authors": [["Unknown"], [''], ['et. al.']],
        "title": ["Title of the paper", "Title not provided"],
        "journal": ["Journal", "Journal not provided"],
        "publisher": ["Publisher", "Publisher not provided"],
        "year": ["Year", "Year not provided"],
        "location": ["Location", "Location not provided"],
        "volume": ["Volume", "Volume not provided"],
        "pages": ["Pages", "Pages not provided"],
        "doi": ["DOI", "DOI not provided"],
        "url": ["URL", "URL not provided"],
    }
    all_invalid_values = ["", "Unknown", "Not provided", "Not specified", "Not available", "N/A", "NA"]
    
    for col, invalid_values in col_invalid_values.items():
        all_data[col] = all_data[col].apply(
            lambda x: pd.NA if x in invalid_values else x
        )

    # Finish cleaning authors column
    all_data["authors"] = all_data["authors"].apply(
        lambda x: [] if not isinstance(x, list) else x
    )
    all_data["authors"] = all_data["authors"].apply(
        lambda x: x if len(x) > 0 else pd.NA
    )
    
    # Finish cleaning dataset
    all_data.replace(all_invalid_values, pd.NA, inplace=True)

    all_data.dropna(
        subset=["title", "authors", "journal", "publisher", "year"], how="all", inplace=True
    )
    
    
    court_case_regex = re.compile(r'\b[A-Z][a-zA-Z., ]+\.? v\. [A-Z][a-zA-Z., ]+')
    all_data = all_data[~all_data["citation"].str.match(court_case_regex, na=False)]
    return all_data



def final_clean(df: pd.DataFrame):
    # Ensure that the authors column is a list of strings
    df['authors'] = df['authors'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Remove duplicate citations (if duplicate within the same regulation)
    non_dups = df.drop_duplicates(subset=['title', 'year', 'docid'], keep='first').copy()
    
    # Load the document and agency information
    with open(Path.home() / r"Box\Fed-Register\all_doc_info.json", 'r') as file:
        all_docs = json.load(file)
    with open(Path.home() / r"Box\Fed-Register\agency_hash.json", 'r') as file:
        agency_hash = json.load(file)
        
        
    # Create a normalized agency column with the highest-level agency name
    for doc in all_docs:
        if 'agencies' in doc:
            for agency in doc['agencies']:
                if 'parent_id' in agency and agency['parent_id'] is not None:
                    agency['name'] = agency_hash.get(str(agency['parent_id']), agency['parent_id'])
                else:
                    agency['name'] = agency['raw_name']
                    
    # Set the agencies and regulation_id_numbers columns
    docid_agency_dict = {doc['document_number']: list(set([agency['name'] for agency in doc['agencies']])) for doc in all_docs}
    docid_rins_dict = {doc['document_number']: doc['regulation_id_numbers'] for doc in all_docs}
    non_dups['agencies'] = non_dups['docid'].map(docid_agency_dict)
    non_dups['regulation_id_numbers'] = non_dups['docid'].map(docid_rins_dict)
    
    # Replace authors with the publisher if authors are missing
    non_dups['regulation_id_numbers'] = non_dups['regulation_id_numbers'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    non_dups['agencies'] = non_dups['agencies'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    non_dups['authors'] = np.where((non_dups['authors'].isnull()) & (~non_dups['publisher'].isnull()), non_dups['publisher'], non_dups['authors'])
    
    # Filling in authors with Publisher if missing author leads to inconsistent datatype (not a list)
    non_dups['authors'] = non_dups['authors'].apply(
        lambda x: f"[\"{x}\"]" if type(x) is str and not ((x.startswith("['") and x.endswith("']")) or (x.startswith("[\"") and x.endswith("\"]"))) else x
    )
    
    
    # Attempt to normalize the names of entities
    with open(r"C:\Users\svens\Box\Fed-Register\entity_hash.json", 'r') as file:
        entity_hash = json.load(file)
        
    non_dups['publisher'] = non_dups['publisher'].replace(entity_hash)
    non_dups['journal'] = non_dups['journal'].replace(entity_hash)
    non_dups["authors"] = non_dups["authors"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    non_dups["authors"] = non_dups["authors"].apply(lambda x: [entity_hash.get(author, author) for author in x] if isinstance(x, list) else x)
    
    # Remove any "et al." from the authors list
    non_dups["authors"] = non_dups["authors"].apply(
        lambda x: [author for author in x if author.lower() != "et al."] if isinstance(x, list) else x
    )
    non_dups["authors"] = non_dups["authors"].apply(
        lambda x: [re.sub(r"\s+et\s+al\.$", "", author, flags=re.IGNORECASE) for author in x] if isinstance(x, list) else x
    )
    
    return non_dups

if __name__ == "__main__":
    directory = Path(r"C:\Users\svens\Box\Fed-Register\final-rule-batch-results-20241102\completed-batches")
    responses = get_responses(directory)
    data = process_responses(responses)
    
    data = final_clean(data)
    
    data.to_csv(
        r"C:\Users\svens\Box\Fed-Register\final_rules_all_data.csv",
        index=False,
    )
