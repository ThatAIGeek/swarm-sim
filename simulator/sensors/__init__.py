"""Сенсори симулятора: що «бачить» кожен дрон на кожному такті.

    neighbors_detector — до 8 найближчих сусідів у радіусі 15 м: Δp, Δv у body frame і маска
    self_sensor        — власна швидкість, напрям на ціль і нормована відстань до неї
    observation        — збирає з сенсорів вхід моделі одного з трьох типів (A, B, C)
"""
from .neighbors_detector import K_MAX, R_DETECT, detect_neighbors, select_neighbors
from .observation import (A_DIM, B_DIM, MASK_COLS, NB_COLS, N_MAX_A, OBS_KINDS, SELF_COLS, build_observation,
                          canonical, observation_A, observation_B, observation_inputs)
from .self_sensor import GOAL_NORM_DIST, sense_self

__all__ = ["K_MAX", "R_DETECT", "detect_neighbors", "select_neighbors", "GOAL_NORM_DIST", "sense_self", "A_DIM",
           "B_DIM", "N_MAX_A", "OBS_KINDS", "NB_COLS", "MASK_COLS", "SELF_COLS", "build_observation", "canonical",
           "observation_A", "observation_B", "observation_inputs"]
