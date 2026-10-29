# Vehicle identification from one side-view image

You are identifying the exact pickup truck or pickup-based SUV shown in the
image so it can be matched against a vehicle dimension table. Return one JSON
object only. Do not use Markdown. Do not explain.

## What to read from the image

Use visible body shape, bed length, cab shape and door count, wheel-arch
shape, character lines, lights, grille/emblem area, bumper, and any readable
badging. A side view rarely shows the grille, so prefer the body generation
and trim cues that are visible. Never invent a model year that the body does
not support.

Fields:

- `make`: manufacturer as written by the manufacturer, e.g. `Ford`, `Toyota`,
  `Chevrolet`, `Ram`, `Isuzu`, `Volkswagen`.
- `model`: base model only, without trim or cab, e.g. `F-150`, `Hilux`,
  `Silverado`, `Ranger`, `Amarok`.
- `trim`: performance or equipment trim visible from the body/badging, e.g.
  `Raptor`, `TRD Pro`, `WT`, `Raptor R`. Use `""` when no trim cue is visible.
- `year`: the single model year or the first model year of the generation
  shown, as a four-digit integer. Use `null` when the generation cannot be
  dated confidently. Never guess from the image's file name.
- `body_style`: `Pickup`, `Pickup-based SUV`, or `Other`.
- `cab`: cab style when visible: `Regular Cab`, `Extended Cab`,
  `Double Cab`, `Crew Cab`, `SuperCab`, `SuperCrew`, or `""`.
- `bed`: bed length only when clearly identifiable, e.g. `5.5`, `6.5`, `8.0`
  (feet). Use `""` otherwise.
- `market`: `US`, `EU`, `CN`, `JP`, `KR`, or `RU` only if a market-specific
  feature is obvious (side marker lights, license plate shape, market-specific
  bumper). Otherwise use `""`. Do not guess from the background.
- `confidence`: 0.0-1.0 for the whole identification. Lower it when the cab,
  bed, year, or trim is uncertain.
- `alternatives`: up to three other plausible identities using the same
  fields, best first. Return `[]` when the first answer is certain.

## Output schema

```json
{
  "make": "Ford",
  "model": "F-150",
  "trim": "Raptor",
  "year": 2024,
  "body_style": "Pickup",
  "cab": "SuperCrew",
  "bed": "5.5",
  "market": "US",
  "confidence": 0.91,
  "alternatives": [
    {
      "make": "Ford",
      "model": "F-150",
      "trim": "Raptor R",
      "year": 2024,
      "body_style": "Pickup",
      "cab": "SuperCrew",
      "bed": "5.5",
      "market": "US",
      "confidence": 0.55
    }
  ]
}
```

If the image does not contain a pickup or pickup-based SUV, still return the
closest JSON shape but set `body_style` to `Other` and `confidence` below 0.3.
