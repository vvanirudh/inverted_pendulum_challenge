# Imitation Learning for Controlling Inverted Pendulum

- Observation Space: Front-facing camera images of the cart-pendulum system (RGB images)
- Control Space: Scalar acceleration applied to the cart (measured in m/s²)
- Expert Data: Trajectories of expert demonstrations, each containing:
  - 5 seconds of data
  - Sampled at 10Hz (every 0.1s)
  - Each sample is a pair of (image, control input)
  - Total of 50 pairs per trajectory

- Number of Expert Trajectories: 100

## Notes
- You can use `test.py` to test your trained controller in the simulated environment.
- `dataset.py` has been provided to load the expert data. All of the raw data is in `pendulum_data/`.
- `env.py` has also been provided for reference, but this is not needed to complete the task.
