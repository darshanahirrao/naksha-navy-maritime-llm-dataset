# Lecture story: from a sighting on the horizon to a decision

This is the narrative that ties the six datasets together. The point of the lecture is
not that a small model knows maritime facts. It is that a small model fine tuned on
*your* data answers questions about *your* area, and a general model does not.

Every fact below is computed from the generated data, so you can ask the same
questions live and get the same answers.

## The cast

Three ships carry the story. Everything else is context.

| Vessel | MMSI | What she is | Why she matters |
| --- | --- | --- | --- |
| MV AL RAYYAN STAR | 470123456 | UAE general cargo, Jebel Ali to west India | The clean sighting. Easy to identify, hard to place in time without data. |
| MT GULF PIONEER | 419876543 | Indian crude tanker, Fujairah to Vadinar shuttle | The loaded versus ballast question. Draught tells you which leg she is on. |
| MV CORAL HORIZON | 353136000 | Panama container feeder, Colombo to Karachi | The fast one. Useful for comparing speed and passage length. |

The recurring unknown is a wooden dhow with no AIS, reported in January, February and
June in the Kutch approaches, and never identified with certainty.

## Scene 1: the sighting

An officer sees a hull on the horizon and reads AL RAYYAN STAR on the bow. No AIS
contact on the display.

Ask the model:

> You sight a vessel on the horizon and read AL RAYYAN STAR on the hull, but she is not
> transmitting AIS. What do you know about her and what should you establish next?

The fine tuned answer gives the registry record, the normal trading pattern, and the
next steps: position by radar, last and next port, port state or VTS record, and any
warning on her track. It also says not to trust the hull name until it is corroborated.

The base model invents a flag, a tonnage and a destination. That is the first thing the
audience sees.

File: `domain_vessel_reports.json`

## Scene 2: where is she coming from

Now the same officer has an AIS contact 60 nm west of Porbandar on 12 March.

> An AIS contact appears about 60 nm west of Porbandar on 12 March 2026. Which tracked
> vessel is it and where is she coming from?

The answer is MT GULF PIONEER, out of Vadinar at 0800Z bound for Fujairah, in ballast
at 5.6 m draught. The draught is the tell. A loaded tanker on that route sits at about
12.5 m.

Follow ups that work well:

> What was the average speed of MT GULF PIONEER from Vadinar to Fujairah on 12 March?
> List the ports MV AL RAYYAN STAR called at between January and August 2026.
> How long did the passage of MV CORAL HORIZON from Kochi to Mundra on 16 February take?

File: `ais_vessel_tracks.json`

## Scene 3: the warning that changes the answer

This is the strongest scene in the lecture. The same contact, now with a hazard on top
of it.

> Was Area Alpha under a live firing warning on 12 March 2026?

Yes. NAVAREA IX 052/2026, in force from 8 March 0600Z to 15 March 1800Z. MT GULF
PIONEER was inside Area Alpha between 0930Z and 1330Z on 12 March.

Then the question that shows what a fine tuned model can actually do:

> Did any tracked vessel pass through Area Alpha while a live firing warning was in
> force?

The answer lists six occasions, each computed from the track and the warning window,
including MV AL RAYYAN STAR inside the box on 27 March within three hours of NAVAREA IX
072/2026 coming into force.

The teaching point: no single dataset answers this. The model has to hold the track
data, the warning window and the area boundary together. That is what the fine tuning
buys you.

File: `navigational_warnings.json`, cross-checked against `ais_vessel_tracks.json`

## Scene 4: the dark contact

Switch to 14 February, 0742Z, 21.53N 069.24E. A radar contact, no AIS.

> How was contact C-12 classified on 14 February 2026 and with what confidence?

Traditional wooden dhow, affiliation unknown, confidence probable, on radar and visual
evidence only. The same profile was reported on 6 January and again on 25 June, which
is what turns three separate sightings into one pattern.

> What are the indicators of a traditional dhow?
> What should be done when a dark contact enters a live firing area?
> Which contacts in the period were classified as suspect?

The suspect list gives four contacts, all fast craft closing on merchant vessels
without AIS, plus one AIS identity mismatch in August. Every one of them was escalated.

Files: `contact_classification.json` and `synthetic_contact_reports.json`

## Scene 5: the pattern across the year

Zoom out. The question an operations room actually asks is what is changing.

> How many merchant transits were recorded in total between January and August 2026?
> Which month had the worst anchorage congestion and where?
> How did tanker traffic change between March and July 2026?
> What does the reporting say about dark contacts and AIS identity mismatches?

The answers give 3,155 transits, a peak in March at 442 and a monsoon low of 331 in
July, worst congestion at Vadinar in March at 28.7 hours average wait, tanker traffic
down about 32 per cent from March to July, and twenty-eight dark contacts across the
period.

File: `shipping_reports.json`

## Scene 6: the fresh scenario

For the last session, hand the officers a position and a time and let them work it
themselves, then check against the model.

> A radar contact is reported at 22.68N 068.95E at 0915Z on 27 March 2026. Which
> tracked vessel does that match, and what was in force at the time?

The full chain is: MV AL RAYYAN STAR, outbound Mundra to Jebel Ali since 0545Z, inside
Area Alpha, with NAVAREA IX 072/2026 live firing in force since 0600Z that morning, and
the correct action is to warn her on VHF channel 16 and report to exercise control.

That is the whole lecture in one question: identify, place, check the warning, decide.

## Base model versus fine tuned model

Run the same five questions before and after training. The pattern is consistent.

| Question | Base model | Fine tuned model |
| --- | --- | --- |
| What is MMSI 470123456? | Invents a ship name and flag | MV AL RAYYAN STAR, IMO 9451234, UAE flag, general cargo |
| Where was MT GULF PIONEER on 12 March? | No usable position | 22.55N 069.05E, course 264, 13.0 knots, Vadinar to Fujairah |
| Was there a warning on that track? | Generic warning text | NAVAREA IX 052/2026, 8 to 15 March, Area Alpha boundary |
| How is a wooden dhow classified? | Generic definition | Unknown affiliation, probable, radar and visual, cross-referenced to two earlier sightings |
| What is the monthly transit trend? | Invents numbers | 442 in March falling to 331 in July, monsoon onset 8 June |

## Mapping to the ten session outline

| Session | Topic from the outline | Which files carry it |
| --- | --- | --- |
| 1 | Meet the LLM, how an LLM reads and reasons | `domain_vessel_reports.json` |
| 2 | Setting up the LLM on the GPU workspace | all six, loaded together |
| 3 | Building the maritime dataset | all six, plus `scripts/generate_dataset.py` |
| 4 | Fine tuning, LoRA | `train_qa.jsonl` |
| 5 | Is the fine tuned model better, tested against the base | `eval_qa.jsonl` |
| 6 | Connecting the model to the current maritime picture, RAG | `ais_vessel_tracks.json` and `navigational_warnings.json` |
| 7 | Putting it together, decision support LLM part 1 | Scenes 1 to 3 |
| 8 | Keeping the human in command, confidence and escalation | `contact_classification.json` |
| 9 | Putting it together, part 2, behaviour out of scope | the dark contact and AIS mismatch cases |
| 10 | Testing the decision support LLM on a fresh scenario | Scene 6 |

## The out of scope answer

Session 9 needs a case where the right answer is "I do not know, refer to an officer".
Use these. They are also in the training data, tagged `out-of-scope`, so the fine tuned
model has seen examples of refusing to guess.

> What is the cargo of the dhow reported on 14 February 2026?
> Which flag does the dhow fly?
> Is the fast boat reported on 21 July 2026 connected to any known group?

The model should say the information is not in the data, state its confidence, and
refer the question to an officer. If it invents a cargo, an owner or a link, that is the
teaching moment: a decision support tool that guesses is worse than one that says it
does not know.

## One caution to state on the day

Everything here is synthetic. Tell the audience that plainly at the start. The value of
the exercise is the method: collect your own data, build the question and answer pairs
from it, fine tune a small model, and measure whether it answers better. The numbers in
this dataset are placeholders for theirs.
