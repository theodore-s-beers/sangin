import csv
import glob
import json
import re
from typing import Optional

from hazm import Normalizer

METER_PATTERN = re.compile(r"^(.+?)\s*\((.+?)\)\s*$")


def extract_meter_parts(meter_string: str) -> Optional[tuple[str, str, str]]:
    match = METER_PATTERN.match(meter_string)

    if match:
        syllables = match.group(1).strip()
        meter_name = match.group(2).strip()

        assert isinstance(syllables, str), "Syllable pattern should be str"
        assert isinstance(meter_name, str), "Meter name should be str"

        assert syllables, "Syllable pattern should not be empty"
        assert meter_name, "Meter name should not be empty"

        base_meter = meter_name.split()[0]
        assert base_meter, "Base meter should not be empty"

        return syllables, meter_name, base_meter
    else:
        if meter_string == "مفتعلن مفتعلن مفتعلن فع":
            syllables = meter_string
            meter_name = "رجز مطوی"
            base_meter = "رجز"
            return syllables, meter_name, base_meter

        if meter_string == "فاعلاتن فاعلن فاعلاتن فاعلن":
            syllables = meter_string
            meter_name = "مدید سالم"
            base_meter = "مدید"
            return syllables, meter_name, base_meter

        if meter_string == "متفاعلن متفاعلن":
            syllables = meter_string
            meter_name = "کامل مربع"
            base_meter = "کامل"
            return syllables, meter_name, base_meter

        if meter_string == "مستفعلتن مستفعلتن":
            return None  # No idea what to call this one

        # I may handle more meters later, but skip for now
        return None


# Many things in this function can throw; I'm not worrying about it
def main() -> None:
    normalizer = Normalizer()
    seen: set[str] = set()

    jsonl_files = glob.glob("data/*.jsonl")

    with open("hemistichs.csv", "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["hemistich", "meter_syllables", "meter_name", "base_meter"])

        for jsonl_file in jsonl_files:
            print(f"Processing {jsonl_file}...")

            with open(jsonl_file, "r", encoding="utf-8") as input_file:
                for line in input_file:
                    if line.strip():
                        poem = json.loads(line)

                        meter_prop = poem["sections"][0]["ganjoorMetre"]
                        if meter_prop is None:
                            print(
                                f"Skipping poem {poem['fullUrl']} due to missing meter"
                            )
                            continue

                        meter_string = meter_prop["rhythm"]

                        meter_parts = extract_meter_parts(meter_string)
                        if meter_parts is None:
                            print(
                                f"Skipping poem {poem['fullUrl']} due to unsupported meter"
                            )
                            continue

                        syllables, meter_name, base_meter = meter_parts

                        for verse in poem["verses"]:
                            normalized = normalizer.normalize(verse["text"])

                            # There are a lot of repeated hemistichs
                            # In the divan of Sa'eb alone, out of a total 141,203 unique hemistichs,
                            # 1,449 are repeated at least once!
                            # I don't have time to dig into this now, but it troubles me
                            # Let's just skip repeats for the time being
                            if normalized in seen:
                                continue

                            seen.add(normalized)
                            writer.writerow(
                                [normalized, syllables, meter_name, base_meter]
                            )

    print(f"Finished processing {len(jsonl_files)} files")
    print(f"Total unique hemistichs: {len(seen)}")


if __name__ == "__main__":
    main()
