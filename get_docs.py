import os
import sys
import json
import time
import logging
import requests
import threading
import concurrent.futures
from pathlib import Path
from collections import namedtuple
from requests.adapters import HTTPAdapter

logging.basicConfig(
    filename= Path.home() / "box/fed-register/logs/get_rules(lt).log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s - %(levelname)s",
)

api_base_url = "https://www.federalregister.gov/api/v1/documents"
params = {
    "per_page": 1000,
    "fields[]": [
        "agencies",
        "title",
        "type",
        "document_number",
        "publication_date",
        "body_html_url",
        "citation",
        "full_text_xml_url",
        "html_url",
        "json_url",
        "pdf_url",
        "raw_text_url",
        "regulation_id_numbers",
        "significant",
        "subtype",
        "topics",
        "volume",
    ],
    "conditions[publication_date][lte]": None,
    "conditions[publication_date][gte]": None,
}

quarters = {
    1: ("01-01", "03-31"),
    2: ("04-01", "06-30"),
    3: ("07-01", "09-30"),
    4: ("10-01", "12-31"),
}


def get_q_documents(session, year):
    running = 0
    all_results = []
    for q in range(1, 5):
        params["conditions[publication_date][gte]"] = f"{year}-{quarters[q][0]}"
        params["conditions[publication_date][lte]"] = f"{year}-{quarters[q][1]}"
        response = session.get(api_base_url, params=params)
        response.raise_for_status()

        data = response.json()
        running += int(data["count"])
        if "results" in data.keys():
            all_results.extend(data["results"])

            # Deal with pagination
            while "next_page_url" in data.keys():
                response = requests.get(data["next_page_url"])
                response.raise_for_status()
                data = response.json()
                all_results.extend(data["results"])

    # Write the results to a JSON file
    print(f"{year}: Found {len(all_results)} Proposed Rules")
    print(f"{year}: Total number of documents: {running}")
    return all_results


def get_all_documents(output_file):
    with requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            futures = [
                executor.submit(get_q_documents, session, year)
                for year in range(1994, 2025)
            ]
            # Wait for all tasks to complete
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

    with open(output_file, "w") as file:
        json.dump(all_results, file, indent=2)


def status_check(folder: Path, tot_count: int):
    while True:
        print(
            f"Files downloaded: {len([file for file in folder.glob('*')])} / {tot_count}"
        )
        time.sleep(60)


def download_xml(session, doc, output_folder, event, lock: threading.Lock):
    if not event.is_set():
        event.wait()
    with lock:
        response = session.get(doc.url)

    match response.status_code:
        case 200:
            with open(f"{output_folder}/{doc.id}.txt", "wb") as file:
                file.write(response.content)
            return True, doc
        case 429:
            with lock:
                try:
                    with open(
                        f"{output_folder.parent}/logs/429_headers.json", "a"
                    ) as file:
                        json.dump(dict(response.headers), file, indent=2)
                except Exception:
                    pass
                logging.warning(f"Rate Limit hit {doc.id}")
            return False, doc
        case _:
            with lock:
                with open(
                    f"{output_folder.parent}/logs/missing_xml_files.txt", "a"
                ) as file:
                    file.write(f"{response.status_code} | {doc.id} | {doc.url}\n")
                logging.warning(f"Failed {doc.id}")
            return True, doc


def get_txt_files(info_file, output_folder, thread_count=16):
    print("Gathering resources...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open specification file
    with open(info_file, "r") as file:
        data = json.load(file)


    # Check if the files have already been downloaded or ar missing
    Doc = namedtuple("Doc", ["id", "url"])
    data = [
        Doc(doc["document_number"], doc["raw_text_url"])
        for doc in data
        if (doc['type'] is not None)
        and (doc["type"].strip().lower() == "rule")
        and (doc["raw_text_url"] is not None)
        and (output_folder / f"{doc['document_number']}.txt").exists() == False
    ]
    
    

    # Start a background thread to give updates on the download status
    start = len([file for file in output_folder.glob('*')])
    threading.Thread(
        target=status_check, args=(output_folder, len(data) + start), daemon=True
    ).start()

    # Download using multithreading
    with requests.Session() as session:
        adapter = HTTPAdapter(
            pool_connections=int(thread_count * 2),
            pool_maxsize=int(thread_count * 1.5),
            max_retries=10,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        with concurrent.futures.ThreadPoolExecutor(thread_count) as executor:
            event = threading.Event()
            event.set()
            lock = threading.Lock()
            futures = [
                executor.submit(download_xml, session, doc, output_folder, event, lock)
                for doc in data
            ]
            for future in concurrent.futures.as_completed(futures):
                if future.exception():
                    with lock:
                        logging.error(f"Error: {future.exception()}")
                    continue
                response, doc = future.result()
                with lock:
                    logging.info(f"Future completed for {doc.id}")
                if not response:
                    with lock:
                        logging.info(f"Waiting for 2.5 min...")
                        print("Waiting for 2.5 min...")
                    event.clear()
                    time.sleep(180)
                    event.set()
                    executor.submit(download_xml, session, doc, event)
                    with lock:
                        logging.info(f"Resuming download...")


if __name__ == "__main__":
    info_file = Path.home() / "box/fed-register/all_doc_info.json"
    output_folder = Path.home() / "box/fed-register/Final-Rule-txts"

    if not info_file.exists():
        print("Getting doc specifications...")
        get_all_documents(info_file)

    if len(sys.argv) > 1:
        get_txt_files(info_file, output_folder, int(sys.argv[1]))
    else:
        get_txt_files(info_file, output_folder)
