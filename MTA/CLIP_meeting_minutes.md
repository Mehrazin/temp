## CLIP Meeting Minutes

- Target deployment moved from **February to April**.
- Channel model will recommend customer journeys; optimizer will apply eligibility, capacity, and business rules.
- Explore a **separate/intermediate output table** since channel-model outputs differ from standard propensity-model outputs.
- **A/B test** (BAU vs. model-based orchestration) remains an option for discussion in Thursday’s workshop.
- Channel cadence: **DM within first 2 weeks**, email flexible, EasyWeb/ATM mostly always-on, Mobile mostly yes/no with possible timing for Mobile Intercept.
- Next steps: prepare DAS output structure, A/B-test proposal, and a slide explaining how CLIP propensity can feed into the channel model.

## Propensity vs. Channel Model

- **CLIP propensity model:** estimates *who is likely to accept the offer*.
- **Channel model:** decides *how to target the customer* — channels, sequence, and journey.
- Two possible designs:
  1. `Propensity → cutoff → Channel model`
  2. `Propensity + customer data → Channel model → target decision + journey`
- The integrated approach could potentially remove hard propensity cutoffs, but this is **not yet a final decision** and will be discussed in the workshop.
