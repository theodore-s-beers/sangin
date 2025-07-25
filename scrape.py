import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from requests.exceptions import RequestException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("masnavi_scraper.log"), logging.StreamHandler()],
)


class MasnaviScraper:
    def __init__(self, output_file: str = "masnavi.jsonl", delay: float = 2.0):
        self.base_url = "https://api.ganjoor.net/api/ganjoor/poem"
        self.params = {
            "catInfo": "false",
            "catPoems": "false",
            "rhymes": "false",
            "recitations": "false",
            "images": "false",
            "songs": "false",
            "comments": "false",
            "verseDetails": "true",
            "navigation": "false",
        }
        self.output_file = Path(output_file)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; MasnaviScraper/1.0; Responsible scraping)"
            }
        )

    def get_poem_url_path(self, poem_number: int) -> str:
        return f"/moulavi/masnavi/daftar1/sh{poem_number}"

    def fetch_poem(self, poem_number: int) -> Optional[dict]:
        url_path = self.get_poem_url_path(poem_number)
        params = self.params.copy()
        params["url"] = url_path

        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Add metadata about the scraping
            data["_scraping_metadata"] = {
                "scraped_at": time.time(),
                "poem_number": poem_number,
                "url_path": url_path,
                "response_size_bytes": len(response.content),
            }

            logging.info(
                f"Successfully fetched poem {poem_number} ({len(response.content)} bytes)"
            )
            return data

        except RequestException as e:
            logging.error(f"Failed to fetch poem {poem_number}: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON for poem {poem_number}: {e}")
            return None

    def save_poem(self, poem_data: dict) -> None:
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(poem_data, ensure_ascii=False) + "\n")

    def load_existing_poems(self) -> set:
        existing = set()

        if self.output_file.exists():
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if "_scraping_metadata" in data:
                                existing.add(data["_scraping_metadata"]["poem_number"])
                logging.info(
                    f"Found {len(existing)} existing poems in {self.output_file}"
                )
            except Exception as e:
                logging.warning(f"Could not load existing poems: {e}")

        return existing

    def scrape_range(self, start: int = 1, end: int = 172, resume: bool = True) -> None:
        existing_poems = self.load_existing_poems() if resume else set()

        total_poems = end - start + 1
        skipped = 0
        scraped = 0
        failed = 0

        logging.info(f"Starting scrape: poems {start}-{end} ({total_poems} total)")
        if existing_poems:
            logging.info(f"Resume mode: will skip {len(existing_poems)} existing poems")

        for poem_num in range(start, end + 1):
            if poem_num in existing_poems:
                skipped += 1
                if skipped % 100 == 0:
                    logging.info(f"Skipped {skipped} existing poems...")
                continue

            poem_data = self.fetch_poem(poem_num)

            if poem_data:
                self.save_poem(poem_data)
                scraped += 1

                if scraped % 50 == 0:
                    logging.info(
                        f"Progress: {scraped} scraped, {failed} failed, {skipped} skipped"
                    )
            else:
                failed += 1

            time.sleep(self.delay)

        logging.info(
            f"Scraping complete: {scraped} scraped, {failed} failed, {skipped} skipped"
        )


def main():
    scraper = MasnaviScraper()  # Using default parameters
    scraper.scrape_range(1, 172)


if __name__ == "__main__":
    main()
