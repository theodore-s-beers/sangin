import csv
import json
import re

METER_PATTERN = re.compile(r"^(.+?)\s*\((.+?)\)\s*$")


def extract_meter_parts(meter_string: str) -> tuple[str, str, str]:
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
        raise ValueError("Meter string is not in the expected format")


def main() -> None:
    with (
        open("saeb_ghazals.jsonl", "r", encoding="utf-8") as input_file,
        open("hemistichs_meters.csv", "w", newline="", encoding="utf-8") as output_file,
    ):
        writer = csv.writer(output_file)
        writer.writerow(["hemistich", "meter_syllables", "meter_name", "base_meter"])

        for line in input_file:
            if line.strip():
                poem = json.loads(line)
                meter_string = poem["sections"][0]["ganjoorMetre"]["rhythm"]
                syllables, meter_name, base_meter = extract_meter_parts(meter_string)

                for verse in poem["verses"]:
                    writer.writerow([verse["text"], syllables, meter_name, base_meter])


if __name__ == "__main__":
    main()
