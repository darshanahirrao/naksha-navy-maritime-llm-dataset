# Naksha Navy - synthetic maritime Q/A dataset

Question and answer training data for the hands-on lecture "Fine tuning a LLM for
maritime domain awareness and decision support", where officers fine tune a small
model (0.6B to 1B parameters) and see it answer maritime questions better than the
untuned base model.

**Everything in this repository is fictional.** No vessel, IMO or MMSI number, owner,
warning, contact, incident or port statistic describes anything real. See the
disclaimer carried inside every JSON file.

## Files

| File | Data type | Examples |
| --- | --- | --- |
| `dataset/domain_vessel_reports.json` | Vessel identity and registry dossiers | 97 |
| `dataset/contact_classification.json` | Classification doctrine, entries and limits of the data | 144 |
| `dataset/navigational_warnings.json` | NAVAREA IX, coastal warnings, notices to mariners | 157 |
| `dataset/ais_vessel_tracks.json` | AIS track, passage and port call records | 519 |
| `dataset/shipping_reports.json` | Monthly and periodic shipping reports | 82 |
| `dataset/synthetic_contact_reports.json` | Synthetic contact reports filed by exercise units | 115 |
| `dataset/train_qa.jsonl` | All of the above, shuffled, 90 per cent split | 1,002 |
| `dataset/eval_qa.jsonl` | Held back 10 per cent split | 112 |

Total: **1,114 question and answer pairs**, covering **1 January to 31 August 2026**.

## Format

Each JSON file is an envelope with metadata and an `examples` array. Each example is
one question and one answer, plus the record it was derived from:

```json
{
  "id": "AIS-0020",
  "topic": "ais_vessel_tracks",
  "question": "When did MV AL RAYYAN STAR arrive at Deendayal (Kandla) after departing Colombo on 2 February 2026?",
  "answer": "MV AL RAYYAN STAR arrived at Deendayal (Kandla) at 0120Z on 6 February 2026, 92 hours after departure from Colombo. The passage was 1190 nm.",
  "source": "AIS passage record, MMSI 470123456, leg 5",
  "tags": ["passage", "arrival"]
}
```

`tags` marks the examples that matter for the lecture, for example `story`,
`horizon-sighting`, `live-firing`, `dark-contact` and `suspect`.

## The world

The dataset describes one coherent fictional picture so a lecture can follow a single
story instead of jumping between unrelated facts.

- **Exercise area**: North Arabian Sea and the approaches to the Gulf of Kutch, with
  four declared areas. Area Alpha is the live firing box, Area Bravo the survey box,
  Area Charlie the cable and drifting hazard box, Area Delta the tanker transfer box.
- **Three tracked vessels**: MV AL RAYYAN STAR (UAE general cargo), MT GULF PIONEER
  (Indian crude tanker on the Fujairah to Vadinar shuttle) and MV CORAL HORIZON
  (Panama container feeder). Between them they make 97 recorded passages.
- **Supporting traffic**: two fishing vessels, one wooden dhow with no AIS, a bulk
  carrier, a second container ship, an Omani general cargo ship, a yacht and a coast
  guard unit.
- **64 warnings** across eight months, including nine live firing warnings, six GNSS
  interference warnings, and survey, cable, dredging, drifting hazard, fishing gear,
  weather and port restriction warnings.
- **34 contact reports**, of which 15 are matched to a tracked vessel from her own
  track data and 19 are independent sightings.

Everything is generated from one script, so the positions, dates, draughts, warnings
and contact reports all agree with each other. If a question asks whether a ship was
inside a live firing area, the answer is computed from her track and the warning
window, not typed by hand.

## Regenerating

```bash
python3 scripts/generate_dataset.py
```

The generator is seeded, so it produces the same dataset every time. Change `SEED` or
the `LEGS`, `WARNINGS`, `HAND_CONTACTS` and `MONTHLY` tables to build a different
exercise, and every derived question and answer updates with it.

## Fine tuning

The JSONL files are already in the two field shape most fine tuning tools accept:

```python
import json

rows = [json.loads(line) for line in open("dataset/train_qa.jsonl")]
```

For a chat format, wrap each pair:

```python
{"messages": [
    {"role": "user", "content": row["question"]},
    {"role": "assistant", "content": row["answer"]},
]}
```

Suggested starting point for a 0.6B to 1B model: 3 epochs, learning rate 2e-4, LoRA
rank 16 to 32 on all attention and MLP projections, sequence length 1024. Hold
`eval_qa.jsonl` back and ask the same questions before and after training to show the
audience the difference.

## Lecture material

`docs/LECTURE_STORY.md` contains the narrative, the scene by scene questions to ask,
and the mapping back to the ten session outline.

## Disclaimer

Synthetic training data. Not for navigation, intelligence or any operational decision.
