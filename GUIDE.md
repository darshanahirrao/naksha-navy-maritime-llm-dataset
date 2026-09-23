# GUIDE: what this is and how to use it

**Read this page first.** Everything else in the folder is data or the tool that made
the data.

## What this is, in one paragraph

This is a set of synthetic maritime training data built to demonstrate one thing to
officers: if you fine tune a small language model on your own maritime data, it answers
questions about your own area better than a general model does. The folder contains
**1,114 question and answer pairs** in six files, one file for each type of maritime
data, covering **1 January to 31 August 2026** in a fictional exercise area in the
North Arabian Sea and the approaches to the Gulf of Kutch. The whole set is generated
from one consistent picture, so a lecture can follow a single story from a sighting on
the horizon to a decision, instead of hopping between unrelated facts.

**Everything in here is invented.** No vessel, IMO or MMSI number, owner, warning,
contact, incident or port statistic describes anything real. It is teaching material,
not intelligence, and it must not be used for navigation or any operational decision.

## What is in the box

| Path | What it is |
| --- | --- |
| `GUIDE.md` | This page |
| `README.md` | Technical notes: format, regeneration, fine tuning recipe |
| `docs/LECTURE_STORY.md` | The lecture narrative, scene by scene, with the questions to ask |
| `dataset/domain_vessel_reports.json` | 97 pairs. Who is this ship: IMO, MMSI, flag, owner, size, registry |
| `dataset/contact_classification.json` | 144 pairs. Classification doctrine, contact entries, and the limits of the data |
| `dataset/navigational_warnings.json` | 157 pairs. NAVAREA IX warnings, coastal warnings, notices to mariners |
| `dataset/ais_vessel_tracks.json` | 519 pairs. Track, passage, port call, position, speed, draught |
| `dataset/shipping_reports.json` | 82 pairs. Monthly and periodic shipping and security reports |
| `dataset/synthetic_contact_reports.json` | 115 pairs. Contact reports filed by exercise units |
| `dataset/train_qa.jsonl` | All 1,114 pairs shuffled, first 90 per cent, for training |
| `dataset/eval_qa.jsonl` | The held back 10 per cent, for measuring whether training worked |
| `scripts/generate_dataset.py` | The generator. Rebuilds the whole set, or builds a new one |

## The one idea to understand

A language model learns from pairs. Give it a question and its correct answer, over and
over, and it starts producing that style of answer. So the work is not "writing a
manual". It is turning existing records into questions and answers.

Each example in the JSON files looks like this:

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

The `source` field is there so an officer can always see which record the answer came
from. The `tags` field marks the examples that matter for the lecture, such as `story`,
`horizon-sighting`, `live-firing`, `dark-contact`, `suspect` and `out-of-scope`.

## The story inside the data

Three ships carry the whole narrative. Everything else is context.

| Vessel | MMSI | What she is |
| --- | --- | --- |
| MV AL RAYYAN STAR | 470123456 | UAE general cargo ship, Jebel Ali to west India |
| MT GULF PIONEER | 419876543 | Indian crude oil tanker, Fujairah to Vadinar shuttle |
| MV CORAL HORIZON | 353136000 | Panama container feeder, Colombo to Kochi, Mundra, Karachi |

The recurring unknown is a wooden dhow with no AIS, reported in the Kutch approaches in
January, February and June and never identified with certainty.

The data holds together because it is all generated from one picture. Positions, dates,
draughts, warnings and contact reports agree with each other. So a question such as
"did any tracked vessel pass through the live firing area while the warning was in
force?" has a real answer, computed from the tracks and the warning windows. The answer
is six occasions, including MT GULF PIONEER inside Area Alpha from 0930Z to 1330Z on
12 March 2026.

That cross-checking question is the heart of the lecture. No single dataset answers it.
The model has to hold the track data, the warning window and the area boundary together
at once, and that is exactly what fine tuning buys you.

## How to use it in the lecture

`docs/LECTURE_STORY.md` has the full script. The short version is six scenes:

1. **The sighting.** A hull is seen on the horizon with a name but no AIS. What do we know about her, and what do we establish next?
2. **Where is she from.** An AIS contact appears west of Porbandar. Which ship is it, where is she coming from, and is she loaded or in ballast?
3. **The warning on her track.** The same contact, now with a live firing warning in force across her route. This is the strongest scene.
4. **The dark contact.** A radar contact with no AIS, classified on radar and visual evidence only, and linked to two earlier sightings.
5. **The pattern across the year.** Transit numbers, anchorage waiting times, tanker traffic and dark contact counts across eight months.
6. **The fresh scenario.** A position and a time are given to the officers. They work it themselves, then check against the model.

Run the same five questions on the untuned base model first, then on the fine tuned
model. The base model invents a flag, a tonnage and a destination. The fine tuned model
answers from the data and cites the record. That contrast is the demonstration.

## How to fine tune on it

The JSONL files are ready to use:

```python
import json

rows = [json.loads(line) for line in open("dataset/train_qa.jsonl")]
```

For a chat style model, wrap each pair:

```python
{"messages": [
    {"role": "user", "content": row["question"]},
    {"role": "assistant", "content": row["answer"]},
]}
```

Starting point for a 0.6B to 1B parameter model: 3 epochs, learning rate 2e-4, LoRA
rank 16 to 32 on all attention and MLP projections, sequence length 1024. Keep
`eval_qa.jsonl` out of training and use it to show the before and after.

## How to change it

```bash
python3 scripts/generate_dataset.py
```

No libraries are needed beyond the Python standard library, and the generator is
seeded, so it rebuilds the identical set. Change the vessel list, the voyage legs, the
warnings or the monthly report tables in that one script, run it again, and every
derived question and answer updates to match. That is the real lesson for the officers:
the method transfers, the numbers are theirs to replace.

## Ground rules

- The data is fictional and is marked that way inside every file.
- It is not for navigation, intelligence or any operational decision.
- Before any external presentation, consider replacing the vessel names and MMSI
  numbers with a set agreed with the audience, so that nothing on screen resembles a
  real hull.

## Where it came from

It was built from a ten session course outline titled "Fine tuning a LLM for maritime
domain awareness and decision support", covering practical fundamentals and dataset
construction on day one, LoRA fine tuning, validation against the base model, RAG over
the live maritime picture and human in the loop guardrails on day two, and a final
decision support build tested on a fresh scenario.
