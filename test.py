from env import PendulumEnv
import torch
import torchvision.transforms as transforms
import time
from model import PendulumController
import numpy as np
import logging
import argparse


def test_controller_in_sim(model, num_episodes=5, max_steps=500):
    """
    Test the trained controller in the simulation environment

    Args:
        model: Trained PendulumController model
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create the environment
    env = PendulumEnv()

    # Define the same transform used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if len(transform.transforms) == 0:
        logging.warning("No transforms applied. Ensure that the model is compatible with the input format.")

    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")

        # Reset environment
        observation = env.reset()

        for step in range(max_steps):
            # Process the observation
            img_tensor = transform(observation).unsqueeze(0).to(device)

            # Get action from policy
            with torch.no_grad():
                action = model(img_tensor).item()

            # Take a step in the environment
            observation, done = env.step(action)

            # Render at a slower pace for visualization
            time.sleep(0.05)

            if done:
                print(f"Episode ended after {step+1} steps. Failed in episode {episode+1}")
                break

        time.sleep(1)  # Pause between episodes

    env.close()

# Add this to your main function to test in simulation
def main_with_simulation(checkpoint_path):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    # Load the trained model
    model = PendulumController()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Test in simulation
    test_controller_in_sim(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Pendulum Controller in simulation.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint.")
    args = parser.parse_args()
    main_with_simulation(args.checkpoint_path)
