# The Coupled Co-Scaling Law - plain-language summary

*A one-page explainer for non-specialists - journalists, funders, and general readers.
Michael Darius Eastwood. Full paper + code: OSF 10.17605/OSF.IO/6C5XB ·
github.com/MichaelDariusEastwood/arc-principle-validation*

## The question

If we build an AI that improves itself - rewrites itself to get smarter, then uses
that to get smarter again - how do we keep it safe? The usual fear is *speed*: it
gets cleverer and cleverer until it runs away from us. The usual reflex is "slow it
down."

## The finding

**Speed is the wrong thing to watch.** What decides safety is whether the part of the
system that keeps it honest grows *at least as fast* as the part that makes it
capable. Picture two runners - "how capable it is" and "how well we can still correct
it." If the correction runner keeps pace, the system stays safe however fast both run.
If correction falls behind, the system becomes dangerous even moving slowly.

There is one twist that makes the result deeper. If the system not only speeds up but
*speeds up its own speeding-up* - a genuine intelligence explosion - then correction
must grow faster still, fast enough to beat the acceleration. The honest headline is
therefore: *the level of speed does not decide the outcome; a contest between two
growth rates does.* Correction's growth rate must beat drift's. We write that as
**beta greater than k**.

The most striking consequence: even a true "hard takeoff," where the machine becomes
almost limitlessly capable in a short time, is controllable - **provided beta > k**.
The explosion's speed does not change that verdict.

## Why it matters

It changes the central safety question from one we cannot enforce - *"how fast is it
growing, and can we pause it?"* - to one we can **measure**: *"is correction keeping
pace - is beta > k?"* That gives regulators and labs a concrete quantity to instrument
and certify, instead of an unenforceable speed limit.

## What is, and is not, established (stated honestly)

- **Proved:** the mathematics of the criterion, checked end-to-end by a runnable
  program that drives a simulated system straight through a finite-time explosion and
  holds its misalignment to zero the whole way. The program confirms the formulae are
  internally consistent - it does **not** yet test the idea against real AI.
- **The decisive next step:** measuring these growth rates on real frontier models. A
  working measurement tool for this has been built and validated; running it across
  today's leading models is the open experiment.
- **No overclaim:** this does not "solve alignment." It identifies the right variable
  to measure and govern, and proves what must hold for a self-improving system to stay
  correctable.

## What is genuinely new here

Much of the underlying mathematics is classical and is credited as such. The original
contributions, stated precisely, are: (1) the **beta > k** criterion as a compact,
measurable test of whether a self-improving system stays correctable; (2) a companion
result that **unblinded safety evaluations can reverse their own conclusions** - so
safety must be measured blind; and (3) a working, blind **measurement instrument** for
running the test on real models.

## Who

Michael Darius Eastwood is an independent researcher in London. He developed this
framework - first set out in his book *Infinite Architects* (2024/2026) and formalised
across a series of open, time-stamped papers - while representing himself in the High
Court and living with diagnosed ADHD and autism. The work is published in full, with
its code and its own adversarial audits, so that anyone can check it or try to break
it.

**The ask, in one line:** fund the experiment that turns a proved criterion into a
measured one - a blind, open, public-good instrument that reports, for every frontier
model, whether its capacity to stay correctable is keeping pace with its capability.
