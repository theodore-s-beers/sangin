import csv
import json
import re

from hazm import Normalizer

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


# Many things in this function can throw; I'm not worrying about it
def main() -> None:
    normalizer = Normalizer()
    seen: set[str] = set()

    with (
        open("saeb_ghazals.jsonl", "r", encoding="utf-8") as input_file,
        open("saeb_meters.csv", "w", newline="", encoding="utf-8") as output_file,
    ):
        writer = csv.writer(output_file)
        writer.writerow(["hemistich", "meter_syllables", "meter_name", "base_meter"])

        for line in input_file:
            if line.strip():
                poem = json.loads(line)

                # I'm surprised the ghazals can be so long, but I guess so...
                assert len(poem["verses"]) >= 8  # i.e., 4 bayts
                assert len(poem["verses"]) <= 102  # i.e., 51 bayts

                meter_string = poem["sections"][0]["ganjoorMetre"]["rhythm"]
                syllables, meter_name, base_meter = extract_meter_parts(meter_string)

                for verse in poem["verses"]:
                    normalized = normalizer.normalize(verse["text"])

                    # There are a lot of repeated hemistichs
                    # Out of a total 141,203 unique hemistichs, 1,449 are repeated at least once
                    # I don't have time to dig into this now, but it troubles me
                    # Let's just skip repeats for the time being
                    if normalized in seen:
                        continue

                    seen.add(normalized)
                    writer.writerow([normalized, syllables, meter_name, base_meter])


if __name__ == "__main__":
    main()
