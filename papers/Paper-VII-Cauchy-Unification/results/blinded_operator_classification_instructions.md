# Blinded Operator Classification Instructions

This packet is the canonical external-review input for Paper VII's empirical cohort.

## Assessor task

For each domain, assign exactly one operator class:

- `multiplicative`: recursive gains compose multiplicatively across scale.
- `additive`: equal additive steps in the independent variable produce equal multiplicative changes in the response.
- `bounded`: the system is constrained by a carrying capacity, finite-site saturation, or hard physical ceiling.

## Blinding rules

- Do not inspect the canonical manifest while classifying.
- Do not use the Paper VII predicted families.
- Use system mechanics, not curve-fit results.
- Record uncertainty when the mechanism is genuinely ambiguous.

## Packet summary

- Domains: 25
- Packet JSON: `blinded_operator_classification_packet.json`
- Response template: `blinded_operator_classification_template.json`

## Submission format

Return one response object per blind ID with:

- `operator_class`
- `confidence`
- `justification`

The packet intentionally neutralizes labels such as `logistic`, `Hill`, or other model-name hints.
