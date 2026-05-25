"""
Detection vocabulary — comprehensive class lists per task.

GroundingDINO performs best with ~20 classes per call,
so we split into batches and merge results.
"""

# ── WalkVLM: outdoor navigation for blind users ──────────────────────────
WALKVLM_CLASSES = [
    # People & vehicles
    "person", "pedestrian", "child", "wheelchair",
    "car", "bus", "truck", "van", "taxi",
    "bicycle", "motorcycle", "scooter", "electric scooter",
    # Traffic infrastructure
    "traffic light", "stop sign", "yield sign", "road sign",
    "crosswalk", "zebra crossing", "traffic cone", "bollard",
    # Road & path
    "sidewalk", "road", "curb", "ramp", "slope",
    "stairs", "step", "escalator", "elevator",
    "bridge", "overpass", "underpass", "tunnel",
    # Obstacles & hazards
    "pole", "lamp post", "fire hydrant", "mailbox",
    "trash can", "dumpster", "construction barrier", "scaffolding",
    "fence", "gate", "wall", "guardrail", "railing",
    "puddle", "pothole", "manhole", "drain", "grate",
    "rock", "debris", "branch", "fallen tree",
    # Street furniture
    "bench", "bus stop", "shelter", "parking meter",
    "bicycle rack", "planter", "flower pot",
    # Nature
    "tree", "bush", "hedge", "grass", "garden",
    # Buildings
    "door", "entrance", "exit", "shop", "store front",
    "window", "awning", "canopy",
]

# ── VisAssist: indoor object recognition & reading ────────────────────────
VISASSIST_CLASSES = [
    # Containers & packaging
    "cup", "mug", "glass", "bottle", "can", "jar",
    "box", "package", "bag", "pouch", "wrapper",
    "container", "tube", "carton", "tin",
    # Food & drink
    "food", "fruit", "vegetable", "bread", "snack",
    "plate", "bowl", "tray", "dish",
    # Text & labels
    "label", "text", "sign", "price tag", "barcode",
    "receipt", "menu", "book", "magazine", "newspaper",
    "screen", "display", "monitor",
    # Medicine & health
    "medicine", "pill bottle", "pill", "syringe",
    "thermometer", "bandage",
    # Household
    "remote", "phone", "charger", "cable", "plug",
    "switch", "button", "knob", "handle", "lever",
    "key", "lock", "door", "drawer", "cabinet",
    "shelf", "rack", "hook", "hanger",
    # Furniture
    "chair", "table", "desk", "sofa", "bed",
    "lamp", "fan", "clock", "mirror",
    # Kitchen
    "stove", "oven", "microwave", "refrigerator", "sink",
    "faucet", "kettle", "toaster", "blender",
    "knife", "spoon", "fork", "chopsticks",
    "cutting board", "pan", "pot", "lid",
    # Clothing & accessories
    "shoe", "hat", "glasses", "watch", "wallet",
    "bag", "backpack", "umbrella",
]

# ── VIA/EgoDex: egocentric hand manipulation ─────────────────────────────
VIA_EGODEX_CLASSES = [
    # Hands & body
    "hand", "left hand", "right hand", "finger", "thumb",
    "wrist", "arm",
    # Containers
    "cup", "mug", "glass", "bottle", "can", "jar",
    "bowl", "plate", "tray", "container", "box",
    "lid", "cap", "cork", "stopper",
    # Tools & utensils
    "spoon", "fork", "knife", "chopsticks",
    "spatula", "ladle", "tongs", "whisk",
    "scissors", "pen", "pencil", "brush",
    "screwdriver", "wrench", "hammer",
    # Kitchen items
    "pan", "pot", "kettle", "cutting board",
    "stove", "burner", "faucet", "sink",
    "sponge", "towel", "cloth", "napkin",
    # Food & ingredients
    "food", "ice", "ice cube", "water", "liquid",
    "egg", "bread", "fruit", "vegetable", "meat",
    # Surfaces & furniture
    "table", "counter", "shelf", "drawer",
    "handle", "knob", "switch", "button",
]

# ── Registry ──────────────────────────────────────────────────────────────
TASK_CLASSES = {
    "walkvlm": WALKVLM_CLASSES,
    "visassist": VISASSIST_CLASSES,
    "via_egodex": VIA_EGODEX_CLASSES,
}


def get_detection_prompts(task: str, batch_size: int = 20) -> list[str]:
    """
    Get detection prompts for a task, split into batches.

    GroundingDINO works best with ~20 classes per prompt.
    Returns a list of prompt strings, each with batch_size classes.

    Args:
        task: "walkvlm", "visassist", or "via_egodex"
        batch_size: max classes per prompt (default 20)

    Returns:
        List of prompt strings like ["person. car. bus. ...", "tree. fence. ..."]
    """
    classes = TASK_CLASSES.get(task, VISASSIST_CLASSES)
    prompts = []
    for i in range(0, len(classes), batch_size):
        batch = classes[i:i + batch_size]
        prompts.append(". ".join(batch) + ".")
    return prompts
