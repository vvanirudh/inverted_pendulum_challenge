import numpy as np
import pygame
import time
from PIL import Image

class PendulumEnv:
    def __init__(self):
        # Physical parameters
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pendulum = 0.1
        self.length = 1.0
        self.total_mass = self.mass_cart + self.mass_pendulum

        # State variables
        self.cart_pos = 0.0
        self.cart_vel = 0.0
        self.pendulum_angle = np.pi # Start upright
        self.pendulum_vel = 0.0

        # Simulation parameters
        self.dt = 0.02  # Simulation time step
        self.max_force = 10.0

        # Rendering
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None

    def reset(self):
        # Reset the state
        self.cart_pos = 0.0
        self.cart_vel = 0.0
        self.pendulum_angle = np.pi + np.random.uniform(-0.2, 0.2)  # Small perturbation
        self.pendulum_vel = 0.0

        return self.get_observation()

    def step(self, action):
        # Apply action (acceleration)
        action = np.clip(action, -self.max_force, self.max_force)

        # Calculate forces
        force = action
        costheta = np.cos(self.pendulum_angle)
        sintheta = np.sin(self.pendulum_angle)

        # Calculate accelerations using the dynamics equations
        temp = (force + self.mass_pendulum * self.length * self.pendulum_vel**2 * sintheta) / self.total_mass
        pendulum_acc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.mass_pendulum * costheta**2 / self.total_mass))
        cart_acc = temp - self.mass_pendulum * self.length * pendulum_acc * costheta / self.total_mass

        # Update state using Euler integration
        self.cart_vel += cart_acc * self.dt
        self.cart_pos += self.cart_vel * self.dt
        self.pendulum_vel += pendulum_acc * self.dt
        self.pendulum_angle += self.pendulum_vel * self.dt

        # Normalize angle to [-pi, pi]
        self.pendulum_angle = ((self.pendulum_angle + np.pi) % (2 * np.pi)) - np.pi

        # Get observation
        observation = self.get_observation()

        # Check termination conditions
        done = False
        if self.cart_pos < -2.5 or self.cart_pos > 2.5:
            done = True

        return observation, done

    def get_observation(self):
        """Return the rendered image as observation"""
        if self.screen is None:
            self.render()

        # Render the current state
        self.render()

        # Convert the Pygame screen to a PIL Image
        assert self.screen is not None
        pygame_img = pygame.surfarray.array3d(self.screen)
        pygame_img = pygame.surfarray.make_surface(pygame_img)
        pil_img = Image.frombytes('RGB', pygame_img.get_size(),
                                  pygame.image.tostring(pygame_img, 'RGB'))

        return pil_img

    def render(self):
        """Render the environment"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Convert cart position to screen coordinates
        cart_x = int(self.screen_width / 2 + self.cart_pos * 50)
        cart_y = int(self.screen_height / 2)

        # Draw the cart
        cart_width, cart_height = 80, 40
        pygame.draw.rect(self.screen, (0, 0, 0),
                         [cart_x - cart_width // 2, cart_y - cart_height // 2,
                          cart_width, cart_height])

        # Draw the pendulum
        pendulum_x = cart_x + int(self.length * 100 * np.sin(self.pendulum_angle))
        pendulum_y = cart_y + int(self.length * 100 * np.cos(self.pendulum_angle))
        pygame.draw.line(self.screen, (0, 0, 255), (cart_x, cart_y), (pendulum_x, pendulum_y), 6)
        pygame.draw.circle(self.screen, (255, 0, 0), (pendulum_x, pendulum_y), 10)

        pygame.display.flip()
        assert self.clock is not None
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
