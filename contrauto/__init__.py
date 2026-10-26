"""ContrAuto: two-layer (discrete contract synthesis + physical execution) controller pipeline."""

import os
import warnings

# contrauto.grounding / sim_iface import pygame too -- silence its import-time banner and
# pkg_resources deprecation warning here as well (see simulator/__init__.py).
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
