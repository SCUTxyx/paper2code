# Gap list: iLQR pendulum

| # | Gap | What's missing | Impact |
|---|---|---|---|
| 1 | Box-aware QP backward pass (Tassa et al. 2012 full formulation) | More complex per-step QP | The clamp variant's O(1) KKT residual at saturated controls is documented & bounded, not eliminated |
| 2 | Real-robot deployment (receding-horizon MPC, actuator dynamics, latency) | Hardware | Control-theory claims verified in simulation only |
| 3 | Learned dynamics models / residual physics | Training data | OUT of scope |
| 4 | Wall-clock / real-time factor claims | Embedded hardware | Not applicable to this repro |
| 5 | Higher-dimensional tasks (arm swing-up, locomotion) | Tuning + compute | Algorithm generalizes; only the pendulum instance verified |
