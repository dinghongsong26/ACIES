"""Shared off-screen GIF capture for BaseGame-backed simulators."""
import os


def render_fsm_gif(game, gif_path, steps, dt, *, every=12, assumption_monitor=None):
    """Execute a finite FSM simulation and save periodically rendered frames as a GIF."""
    from PIL import Image
    import pygame
    from contrauto.sim_iface import run_fsm

    frames = []
    tobytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring

    def capture(step, _world):
        if step % every == 0:
            game.draw()
            image = Image.frombytes("RGB", game.screen.get_size(), tobytes(game.screen, "RGB"))
            frames.append(image)

    try:
        metrics = run_fsm(game, game.controllers, steps, dt, on_step=capture,
                          assumption_monitor=assumption_monitor)
        if frames:
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                           duration=int(every * dt * 1000), loop=0, optimize=True)
        return metrics
    finally:
        pygame.quit()
