# Lumen AI VP Presentation — Slide Messages and Speaker Talk Track

## Purpose of the Deck

This 4-slide deck is designed for a VP / senior business audience. The goal is to keep the story simple, business-focused, and non-technical:

1. Why Lumen AI matters.
2. How we learn from historical customer journeys.
3. How we turn that learning into practical channel recommendations.
4. What changes under Lumen AI and what impact we expect.

The recommended structure is **headline first, proof later**: mention the expected result early at a high level, then use the final slide to show the evidence and business impact.

---

# Slide 1 — Akber

## Slide Title

**Lumen AI: From Attribution to Channel Decisioning**

## Speaker

**Akber**

## Main Message

Lumen AI is not just an attribution exercise. It is a framework that helps us learn from historical customer journeys and then use that learning to guide future channel decisions.

## What Akber Should Say

Today, we want to walk through Lumen AI, which started as a multi-touch attribution initiative but has evolved into a broader channel decisioning framework.

The business problem is simple: today, channel assignment is still largely driven by predefined rules. For example, certain customer groups may be assigned to Leads, email, direct mail, or other channels based on broad rules or propensity bands. That approach works, but it does not fully use the customer journey evidence we already have.

Lumen AI is designed to add a data-driven layer on top of that process. It has two connected lenses.

The first is a **diagnostic lens**, where we learn from historical touchpoints and understand which channels contributed to conversion.

The second is a **decision lens**, where we use that learning to recommend the best controllable channel for each customer, while respecting business constraints such as channel capacity.

The simple way to position this is:

> **PRISM helps us decide what to offer. Lumen AI helps us decide how to reach the customer.**

In historical simulation, we see that this approach produces a different channel mix than BAU and suggests model-estimated uplift versus the current approach. We will walk through how we got there and where the result should be validated next.

## Transition to Nazanin

Before we talk about recommendations, Nazanin will walk through the historical learning piece — what we are using to understand channel impact from past customer journeys.

## One-Line Version

Lumen AI connects attribution and decisioning: it learns from historical journeys and uses that learning to guide future channel decisions.

---

# Slide 2 — Nazanin

## Slide Title

**Diagnostic Lens: Understanding Historical Channel Impact**

## Speaker

**Nazanin**

## Main Message

Diagnostic modeling helps explain what happened historically, especially because raw channel performance can be misleading when customers are not randomly assigned to channels.

## What Nazanin Should Say

This slide explains the first lens of Lumen AI: the diagnostic or historical learning lens.

The key point is that we cannot simply compare raw conversion rates across channels and conclude that one channel is better than another. Customers are not randomly assigned to channels. In many cases, higher-propensity customers are more likely to receive expensive or targeted treatments, such as Leads. So a channel may look strong partly because it was assigned to stronger customers.

To address this, we bring together three types of information.

First, we use observed touchpoints across the customer journey. This can include email, Leads, paid search, SEO, display, web, mobile, branch, and other historical interactions.

Second, we include customer context, such as PRISM scores, existing card indicators, and journey history.

Third, we connect those inputs to the outcome signal, which in this case is new credit card acquisition.

The purpose is to build a more credible view of channel impact. Instead of asking only, “Which channel had the highest raw response rate?”, we ask, “After accounting for customer context and selection effects, which channels appear to contribute to conversion?”

One important distinction is that diagnostic modeling can use the full observed journey, including channels like paid search, SEO, display, and web behavior. These are valuable signals for understanding the journey, even if they are not always channels that we can directly assign to a customer in a future campaign.

So the takeaway from this slide is: diagnostic modeling helps us learn from historical journeys before we decide where TD can actually act.

## Transition to Vandad

Once we have that historical learning, the next question is how to turn it into a practical recommendation. Vandad will walk through that decisioning lens.

## One-Line Version

Diagnostic modeling helps us understand historical channel impact more fairly, especially because raw channel performance is biased by how customers were selected.

---

# Slide 3 — Vandad

## Slide Title

**Decision Lens: Recommend Channels We Can Control**

## Speaker

**Vandad**

## Main Message

The predictive and optimization part should focus only on channels TD can actually assign or control at the customer level.

## What Vandad Should Say

This slide shows how we move from historical learning to practical channel recommendations.

A key design choice is separating **signals we observe** from **actions we can choose**.

Some touchpoints are very useful for diagnosis and learning. For example, paid search, SEO, display, web behavior, email engagement, and customer reactions all help us understand the customer journey. But not all of these are direct customer-level actions that we can assign in the same way.

For example, paid search is useful as a historical signal, but it does not naturally work as a next-best outbound channel for a specific customer in the same way as email, direct mail, Leads, or a call list.

So for predictive optimization, we focus on **controllable customer-level actions**. These are channels where TD can make a practical assignment, such as email, direct mail, Leads, phone, mobile or app placement, and branch prompts, depending on operational availability.

The recommendation process has three steps.

First, for each customer, we score the candidate channels. The model estimates the value or expected lift of each possible channel option.

Second, we optimize the allocation. This means we select the best channel for each customer while respecting constraints. For example, if Leads can only support 30,000 customers, the optimizer can enforce that capacity.

Third, we output the recommendation. That can include the best channel per customer, the constrained channel plan, and the expected impact versus BAU.

The main point is that the optimizer recommends only channels we can practically assign, while the broader diagnostic signals still help inform the learning behind the model.

## Transition to Mehrazin

Now that we have the diagnostic and decisioning framework, Mehrazin will show what changes when we compare BAU with the optimized channel allocation in historical simulation.

## One-Line Version

The optimizer focuses on channels we can actually control, scoring each customer-channel option and applying business constraints like Leads capacity.

---

# Slide 4 — Mehrazin

## Slide Title

**What Changes Under Lumen AI**

## Speaker

**Mehrazin**

## Main Message

Lumen AI changes the controllable-channel mix and estimates incremental conversion opportunity versus BAU, but the result should be framed as model-estimated and subject to live validation.

## What Mehrazin Should Say

This slide brings the story together and shows what changes when we apply Lumen AI to historical data.

On the left, we compare the current BAU channel mix with the optimizer-recommended channel mix. The point is not just that the percentages are different. The point is that the optimizer is making customer-level decisions based on estimated channel value, rather than relying only on broad rules.

The unconstrained scenario shows what the model would choose if there were no operational capacity limits. This is useful because it shows the full model-preferred allocation.

The constrained scenario is more practical. It applies business constraints, such as a Leads capacity cap, so the output is closer to something the business could actually use.

On the right, we translate the optimized allocation into expected conversion impact. The BAU bar represents the observed historical baseline. The optimizer bars represent model-estimated expected conversion under the recommended allocation.

The key message is that, in historical simulation, Lumen AI recommends a different and more personalized controllable-channel mix. Based on model-estimated lift, this suggests an expected improvement versus BAU.

However, we should be careful with the wording. This is not yet a live A/B test result. We should not say the model has already proven uplift. The right wording is:

> **The historical simulation suggests expected uplift, and the next step would be live validation through deployment or a champion/challenger test.**

So the takeaway is: Lumen AI gives us a practical way to move from historical attribution to customer-level channel decisioning, with a measurable expected business impact and a clear path to validation.

## Closing Statement

Our recommendation is to continue validating this framework with business partners, focus the predictive decisioning on controllable channels, and use the diagnostic lens to keep learning from the full customer journey.

## One-Line Version

In historical simulation, Lumen AI changes the channel mix and suggests model-estimated conversion uplift versus BAU, which should be validated live.

---

# Full Presentation Flow

## Recommended Speaker Order

1. **Akber** — executive framing and business problem.
2. **Nazanin** — diagnostic learning from historical journeys.
3. **Vandad** — controllable-channel recommendation and optimization.
4. **Mehrazin** — results, expected impact, and validation path.

## Recommended Timing

| Slide | Speaker | Suggested Time |
|---|---:|---:|
| Slide 1 | Akber | 1.5–2 minutes |
| Slide 2 | Nazanin | 1.5–2 minutes |
| Slide 3 | Vandad | 2 minutes |
| Slide 4 | Mehrazin | 2–3 minutes |

Total recommended presentation time: **7–9 minutes**, leaving time for questions.

---

# Executive-Level Story in One Paragraph

Lumen AI helps move channel strategy from broad rule-based assignment toward customer-level, data-driven channel decisioning. It first uses a diagnostic lens to understand historical customer journeys and channel impact while accounting for selection bias. It then uses a decisioning lens to recommend controllable channels — such as email, direct mail, Leads, phone, mobile, or branch — where TD can actually make a customer-level assignment. In historical simulation, this produces a different channel mix than BAU and suggests model-estimated conversion uplift, subject to live validation.

---

# Important Wording to Use

Use this wording:

> **Model-estimated expected uplift**

> **Historical simulation**

> **Subject to live validation**

> **Controllable customer-level channels**

> **Diagnostic lens for historical learning**

> **Decision lens for future channel recommendations**

---

# Wording to Avoid

Avoid saying:

> The model proved uplift.

> The optimizer increased conversion by X%.

> Paid search is recommended as a direct outbound channel.

> This is a fully causal result.

> This replaces business decision-making.

Instead, say:

> The model suggests expected uplift in historical simulation.

> The optimizer recommends controllable channels where TD can make customer-level assignments.

> Diagnostic signals such as paid search help us understand the journey, even if they are not direct next-best-channel actions.

> A live champion/challenger or A/B test would be needed to validate realized impact.

---

# Q&A Prep

## If asked: “Is this already proven?”

Suggested answer:

Not yet as a live business outcome. The result is based on historical simulation and model-estimated lift. It gives us a strong directional case, but realized uplift should be validated through a live champion/challenger test or phased deployment.

## If asked: “Why not use raw conversion rates?”

Suggested answer:

Raw conversion rates can be misleading because customers are not randomly assigned to channels. For example, high-propensity customers may be more likely to receive Leads, so Leads can look strong partly because of customer selection. The diagnostic model helps account for customer context and gives a more credible view of channel impact.

## If asked: “Can the model recommend paid search?”

Suggested answer:

Paid search is useful as a diagnostic signal because it helps us understand the historical customer journey. But it is not naturally a customer-level outbound action in the same way as email, direct mail, Leads, or phone. For optimization, we focus on channels TD can directly assign or control at the customer level.

## If asked: “What is the next step?”

Suggested answer:

The next step is to validate the framework with business partners, align on which channels are operationally controllable, include real business constraints, and test the recommendations through live deployment or a champion/challenger setup.

## If asked: “How does this relate to PRISM?”

Suggested answer:

PRISM helps determine customer and product propensity — essentially what to offer and to whom. Lumen AI complements that by focusing on how to reach the customer through the right channel.

