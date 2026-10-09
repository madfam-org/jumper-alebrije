"""Alebrije: the painted fantastical creatures of Oaxaca and Mexico City folk art.

Rosa mexicano shell with yellow, turquoise and black: rings of dots, a sun
medallion, a spiny crest, a zigzag crown, feathered wings and big yellow eyes
on stalks. Lacquered rainbow legs. Released as v1.0.0; its map keeps the id it
shipped with.
"""

from jumperkit import parts as P
from jumperkit.character import Character

PALETTE = {
    "rosa": "#E4007C",      # rosa mexicano
    "turquesa": "#00B3A6",
    "amarillo": "#FFC20E",
    "naranja": "#FF6A13",
    "cobalto": "#1F4FD8",
    "lima": "#7DC242",
    "morado": "#6A2FB8",
    "blanco": "#FFF6E0",
    "negro": "#141414",
}


def decorate(d: P.Decor, shell: P.Shell) -> None:
    P.dot_rings(d, shell, [
        (0.030, 0.026, 8, 0.0062, ["amarillo", "turquesa"], None),
        (0.052, 0.046, 12, 0.0072, ["amarillo"], "negro"),
        (0.074, 0.066, 18, 0.0050, ["turquesa"], None),
    ])
    P.medallion(d, shell, "amarillo", "negro", "turquesa")
    P.crest(d, shell, ["amarillo", "turquesa"])
    P.zigzag_crown(d, shell, ["turquesa", "amarillo"])
    P.wings(d, shell, ["turquesa", "amarillo", "rosa", "turquesa", "amarillo", "negro"],
            root_color="amarillo")
    P.eyestalks(d, shell, stalk="turquesa", eye="amarillo", pupil="negro", lid="turquesa")


CHARACTER = Character(
    id="alebrije",
    title="Alebrije Jumper",
    description="Mexican folk-art alebrije: pink shell with dots, crest, wings and yellow eyestalks.",
    palette=PALETTE,
    shell_slots=("rosa", "amarillo", "turquesa", "negro"),
    limb_colors={
        "base_link": "morado",
        # front legs are arms
        "shoulder_link": "amarillo",
        "upper_arm_link": "turquesa",
        "forearm_link": "naranja",
        "palm_link": "cobalto",
        "palm_pad_f_link": "amarillo",
        "palm_pad_b_link": "amarillo",
        "finger_link": "lima",
        "finger_tip_link": "amarillo",
        "finger_grip_insert_link": "blanco",
        "palm_grip_insert_link": "blanco",
        # middle and rear legs
        "hip_link": "amarillo",
        "thigh_link": "turquesa",
        "calf_link": "naranja",
        "foot_tip_link": "lima",
    },
    middle_swap={"turquesa": "cobalto", "naranja": "lima", "lima": "amarillo"},
    decorate=decorate,
    map_id="dia-de-muertos-plaza",
)
