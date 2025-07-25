import numpy as np
import pygame
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

try:
    from IPython.display import display, clear_output
    from IPython import get_ipython

    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

PERTURBATION_RAD = 0.2


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
        self.pendulum_angle = 0.0  # Start upright (0 = upright, π = hanging down)
        self.pendulum_vel = 0.0

        # Simulation parameters
        self.dt = 0.02  # Simulation time step
        self.max_force = 10.0

        # Rendering
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.use_jupyter = self._detect_jupyter()
        self.display_handle = None

    def _detect_jupyter(self):
        """Detect if we're running in a Jupyter notebook"""
        if not JUPYTER_AVAILABLE:
            return False
        try:
            shell = get_ipython().__class__.__name__
            return shell in ["ZMQInteractiveShell", "Shell"]
        except (NameError, AttributeError):
            return False

    def reset(self):
        # Reset the state
        self.cart_pos = 0.0
        self.cart_vel = 0.0
        self.pendulum_angle = np.random.uniform(
            -PERTURBATION_RAD, PERTURBATION_RAD
        )  # Small perturbation around upright (0)
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
        temp = (
            force + self.mass_pendulum * self.length * self.pendulum_vel**2 * sintheta
        ) / self.total_mass
        pendulum_acc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.mass_pendulum * costheta**2 / self.total_mass)
        )
        cart_acc = (
            temp
            - self.mass_pendulum
            * self.length
            * pendulum_acc
            * costheta
            / self.total_mass
        )

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
        if abs(self.pendulum_angle) > np.pi / 2:
            done = True

        return observation, done

    def get_observation(self):
        """Return the rendered image as observation"""
        if self.use_jupyter:
            return self._get_observation_jupyter()
        else:
            return self._get_observation_pygame()

    def _get_observation_pygame(self):
        """Get observation using pygame rendering"""
        if self.screen is None:
            self.render()

        # Render the current state
        self.render()

        # Convert the Pygame screen to a PIL Image
        assert self.screen is not None
        pygame_img = pygame.surfarray.array3d(self.screen)
        pygame_img = pygame.surfarray.make_surface(pygame_img)
        pil_img = Image.frombytes(
            "RGB", pygame_img.get_size(), pygame.image.tostring(pygame_img, "RGB")
        )

        return pil_img

    def _get_observation_jupyter(self):
        """Get observation using matplotlib rendering"""
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(8, 5))

        # Set up the coordinate system (same as _render_jupyter)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Position")
        ax.set_ylabel("Height")
        ax.set_title("Inverted Pendulum")

        # Draw the track
        ax.axhline(y=0, color="gray", linewidth=2, alpha=0.5)

        # Cart parameters
        cart_width = 0.3
        cart_height = 0.2

        # Draw the cart
        cart_rect = patches.Rectangle(
            (self.cart_pos - cart_width / 2, -cart_height / 2),
            cart_width,
            cart_height,
            linewidth=2,
            edgecolor="black",
            facecolor="lightblue",
        )
        ax.add_patch(cart_rect)

        # Draw the pendulum
        pendulum_x = self.cart_pos + self.length * np.sin(self.pendulum_angle)
        pendulum_y = self.length * np.cos(self.pendulum_angle)

        # Pendulum rod
        ax.plot(
            [self.cart_pos, pendulum_x],
            [0, pendulum_y],
            "b-",
            linewidth=3,
            label="Pendulum rod",
        )

        # Pendulum mass
        ax.plot(pendulum_x, pendulum_y, "ro", markersize=10, label="Mass")

        # Cart center
        ax.plot(self.cart_pos, 0, "ks", markersize=6)

        plt.tight_layout()

        # Convert matplotlib figure to PIL Image
        buf = BytesIO()
        fig.savefig(buf, format="jpg", dpi=100, bbox_inches="tight")
        buf.seek(0)
        pil_img = Image.open(buf)

        plt.close(fig)
        return pil_img

    def render(self):
        """Render the environment"""
        if self.use_jupyter:
            self._render_jupyter()
        else:
            self._render_pygame()

    def _render_pygame(self):
        """Render using pygame for regular Python environments"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Convert cart position to screen coordinates
        cart_x = int(self.screen_width / 2 + self.cart_pos * 50)
        cart_y = int(self.screen_height / 2)

        # Draw the cart
        cart_width, cart_height = 80, 40
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            [
                cart_x - cart_width // 2,
                cart_y - cart_height // 2,
                cart_width,
                cart_height,
            ],
        )

        # Draw the pendulum
        pendulum_x = cart_x + int(self.length * 100 * np.sin(self.pendulum_angle))
        pendulum_y = cart_y - int(self.length * 100 * np.cos(self.pendulum_angle))
        pygame.draw.line(
            self.screen, (0, 0, 255), (cart_x, cart_y), (pendulum_x, pendulum_y), 6
        )
        pygame.draw.circle(self.screen, (255, 0, 0), (pendulum_x, pendulum_y), 10)

        pygame.display.flip()
        assert self.clock is not None
        self.clock.tick(60)

    def _render_jupyter(self):
        """Render using matplotlib for Jupyter notebook environments"""
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(8, 5))

        # Set up the coordinate system
        ax.set_xlim(-4, 4)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Position")
        ax.set_ylabel("Height")
        ax.set_title("Inverted Pendulum")

        # Draw the track
        ax.axhline(y=0, color="gray", linewidth=2, alpha=0.5)

        # Cart parameters
        cart_width = 0.3
        cart_height = 0.2

        # Draw the cart
        cart_rect = patches.Rectangle(
            (self.cart_pos - cart_width / 2, -cart_height / 2),
            cart_width,
            cart_height,
            linewidth=2,
            edgecolor="black",
            facecolor="lightblue",
        )
        ax.add_patch(cart_rect)

        # Draw the pendulum
        pendulum_x = self.cart_pos + self.length * np.sin(self.pendulum_angle)
        pendulum_y = self.length * np.cos(self.pendulum_angle)

        # Pendulum rod
        ax.plot(
            [self.cart_pos, pendulum_x],
            [0, pendulum_y],
            "b-",
            linewidth=3,
            label="Pendulum rod",
        )

        # Pendulum mass
        ax.plot(pendulum_x, pendulum_y, "ro", markersize=10, label="Mass")

        # Cart center
        ax.plot(self.cart_pos, 0, "ks", markersize=6)

        plt.tight_layout()

        # Display frame without clearing text output
        if JUPYTER_AVAILABLE:
            if self.display_handle is None:
                self.display_handle = display(fig, display_id=True)
            else:
                self.display_handle.update(fig)

        plt.close(fig)

    def close(self):
        if self.screen is not None:
            pygame.quit()
