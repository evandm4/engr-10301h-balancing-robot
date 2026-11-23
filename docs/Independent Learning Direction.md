**General NN / ML**
- 3Blue1Brown, "Neural Networks", chapters 1–4 (about 1 h): what a network is, gradient descent, backpropagation. Covers gap 1 and part of gap 2.
- OpenAI Spinning Up, Part 1 ("Key Concepts", "Kinds of RL Algorithms", "Intro to Policy Optimization"; about 2–3 h). Covers gaps 1 and 4.
- Henderson et al., "Deep Reinforcement Learning that Matters" (2018), plus the summary of Agarwal et al., "Deep RL at the Edge of the Statistical Precipice" (2021) (about 1 h). These explain why seeds and confidence intervals matter in RL and how to report them, so they feed gap 3 and your results section directly.
- Later, when PPO is being built: "The 37 Implementation Details of PPO" (Huang et al., ICLR blog track, 2022). It's the reference behind the plan's PPO section.
**Robust Control**
- H-infinity design: you describe the plant, then choose the controller that minimizes the worst-case amplification from disturbances (pushes, noise) to errors, across every frequency.
- Mu-synthesis: this goes further. You state your uncertainty explicitly (body mass ±15%, latency 5–20 ms), and the design guarantees stability and performance for every robot in that range.

Your ML approach gets robustness by testing against many randomized robots. Robust control gets it from mathematical guarantees about a stated range. Comparing the two is a natural question for your paper.

To get started:

- Brian Douglas's "Robust Control" videos (MATLAB Tech Talks), about an hour, for intuition.
- Skogestad & Postlethwaite, Multivariable Feedback Control: the standard textbook, if you want depth.