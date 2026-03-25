"""
CS2 game utilities — weapon classification, economy analysis, and related helpers.

Shared constants and functions used across training scripts and diagnostics.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Weapon sets
# ---------------------------------------------------------------------------

WEAPON_PRIMARY = {
    "AK-47", "M4A4", "M4A1-S", "AWP", "SSG 08", "Scout",
    "Galil AR", "FAMAS", "SG 553", "AUG",
    "MAC-10", "MP9", "MP7", "UMP-45", "PP-Bizon", "P90",
    "Nova", "XM1014", "Sawed-Off", "MAG-7",
    "M249", "Negev",
}

WEAPON_SECONDARY = {
    "Glock-18", "USP-S", "P2000", "P250", "Five-SeveN", "Tec-9",
    "CZ75-Auto", "Dual Berettas", "Desert Eagle", "R8 Revolver",
}

UTILITY_ITEMS = {
    "Smoke Grenade", "Flashbang", "Molotov", "Incendiary Grenade",
    "HE Grenade", "High Explosive Grenade", "Decoy Grenade",
}

BOMB_ITEM = "C4 Explosive"

# Approximate weapon values for economy classification (USD)
WEAPON_VALUES = {
    "AK-47": 2700, "M4A4": 3100, "M4A1-S": 2900, "AWP": 4750,
    "SSG 08": 1700, "Galil AR": 1800, "FAMAS": 2050, "SG 553": 3000,
    "AUG": 3300, "MAC-10": 1050, "MP9": 1250, "MP7": 1500,
    "UMP-45": 1200, "PP-Bizon": 1400, "P90": 2350, "Nova": 1050,
    "XM1014": 2000, "Sawed-Off": 1100, "MAG-7": 1300, "M249": 5200,
    "Negev": 1700, "Glock-18": 200, "USP-S": 200, "P2000": 200,
    "P250": 300, "Five-SeveN": 500, "Tec-9": 500, "CZ75-Auto": 500,
    "Dual Berettas": 300, "Desert Eagle": 700, "R8 Revolver": 600,
}

ARMOR_COST = 650   # vest
HELMET_COST = 350   # helmet addon

# Weapon class mapping for sparsity analysis (D023)
WEAPON_CLASS_MAP = {
    # rifle
    "AK-47": "rifle", "M4A4": "rifle", "M4A1-S": "rifle",
    "AUG": "rifle", "SG 553": "rifle",
    # awp
    "AWP": "awp",
    # force_rifle
    "Galil AR": "force_rifle", "FAMAS": "force_rifle",
    # smg_shotgun
    "MAC-10": "smg_shotgun", "MP9": "smg_shotgun", "MP7": "smg_shotgun",
    "UMP-45": "smg_shotgun", "PP-Bizon": "smg_shotgun", "P90": "smg_shotgun",
    "Nova": "smg_shotgun", "XM1014": "smg_shotgun", "Sawed-Off": "smg_shotgun",
    "MAG-7": "smg_shotgun", "M249": "smg_shotgun", "Negev": "smg_shotgun",
    # force_angle
    "Desert Eagle": "force_angle", "R8 Revolver": "force_angle",
    "SSG 08": "force_angle",
    # pistol
    "USP-S": "pistol", "Glock-18": "pistol", "P2000": "pistol",
    "P250": "pistol", "Five-SeveN": "pistol", "Tec-9": "pistol",
    "CZ75-Auto": "pistol", "Dual Berettas": "pistol",
}


# ---------------------------------------------------------------------------
# Weapon classification
# ---------------------------------------------------------------------------

def classify_weapon(item: str) -> str:
    """Classify an inventory item as primary/secondary/utility/melee/bomb."""
    if item in WEAPON_PRIMARY:
        return "primary"
    if item in WEAPON_SECONDARY:
        return "secondary"
    if item in UTILITY_ITEMS:
        return "utility"
    if "C4" in item:
        return "bomb"
    if "Knife" in item or "Bayonet" in item or "Dagger" in item:
        return "melee"
    # Fuzzy fallback
    for w in WEAPON_PRIMARY:
        if w in item or item in w:
            return "primary"
    for w in WEAPON_SECONDARY:
        if w in item or item in w:
            return "secondary"
    for u in UTILITY_ITEMS:
        if u in item or item in u:
            return "utility"
    return "melee"


def classify_weapon_class(inventory: list | None) -> str:
    """Get weapon class of best weapon in inventory (for sparsity bucketing)."""
    if not inventory:
        return "pistol"
    priority = {"awp": 6, "rifle": 5, "force_rifle": 4,
                "smg_shotgun": 3, "force_angle": 2, "pistol": 1}
    best_class = "pistol"
    best_priority = 0
    for item in inventory:
        item_str = str(item)
        wc = WEAPON_CLASS_MAP.get(item_str)
        if wc and priority.get(wc, 0) > best_priority:
            best_class = wc
            best_priority = priority[wc]
    return best_class


def parse_inventory(inventory: list | None) -> dict:
    """Parse inventory list into weapon_primary, weapon_secondary, utility."""
    result: dict = {"weapon_primary": None, "weapon_secondary": None, "utility": []}
    if not inventory:
        return result

    for item in inventory:
        item_str = str(item)
        cat = classify_weapon(item_str)
        if cat == "primary" and result["weapon_primary"] is None:
            result["weapon_primary"] = item_str
        elif cat == "secondary" and result["weapon_secondary"] is None:
            result["weapon_secondary"] = item_str
        elif cat == "utility":
            result["utility"].append(item_str)

    return result


# ---------------------------------------------------------------------------
# Economy classification
# ---------------------------------------------------------------------------

def estimate_equipment_value(player: dict) -> int:
    """Estimate a player's equipment value from inventory and armor."""
    value = 0
    inventory = player.get("inventory") or []
    for item in inventory:
        item_str = str(item)
        value += WEAPON_VALUES.get(item_str, 0)
    if (player.get("armor") or 0) > 0:
        value += ARMOR_COST
        if player.get("has_helmet"):
            value += HELMET_COST
    return value


def estimate_team_equip(players: list[dict], side: str) -> float:
    """Average equipment value for alive players on a side."""
    team = [p for p in players if p.get("side", "").lower() == side.lower()
            and (p.get("health") or 0) > 0]
    if not team:
        return 0
    total = sum(estimate_equipment_value(p) for p in team)
    return total / len(team)


def classify_buy(avg_equip: float) -> str:
    """Classify buy level from average equipment value (short labels)."""
    if avg_equip >= 3500:
        return "full"
    elif avg_equip >= 2000:
        return "half"
    elif avg_equip >= 1000:
        return "force"
    else:
        return "eco"


def classify_team_economy(players: list[dict], side: str) -> tuple[str, int]:
    """Classify a team's buy as full-buy/half-buy/eco/force-buy with total value.

    Returns (buy_type, total_equipment_value).
    """
    team = [p for p in players if (p.get("side") or "").lower() == side.lower()]
    if not team:
        return "unknown", 0

    total_value = sum(estimate_equipment_value(p) for p in team)
    alive_count = sum(1 for p in team if (p.get("health") or 0) > 0)

    if alive_count == 0:
        return "eliminated", 0

    avg_value = total_value / alive_count

    if avg_value >= 3500:
        return "full-buy", total_value
    elif avg_value >= 2000:
        return "half-buy", total_value
    elif avg_value >= 1000:
        return "force-buy", total_value
    else:
        return "eco", total_value


def economy_matchup(t_buy: str, ct_buy: str) -> str:
    """Classify economy matchup into ~6 categories."""
    if t_buy == ct_buy:
        return f"mirror_{t_buy}"
    strength = {"eco": 0, "force": 1, "half": 2, "full": 3}
    if strength.get(t_buy, 0) < strength.get(ct_buy, 0):
        return f"{t_buy}_vs_{ct_buy}"
    else:
        return f"{ct_buy}_vs_{t_buy}"
