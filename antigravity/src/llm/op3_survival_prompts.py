OP3_SURVIVAL_PROMPT = """You are analyzing a drone image crop from an afforestation site after OP3 (weeding).
In OP3, a 1m diameter soil circle is cleared around each surviving sapling. A surviving sapling usually appears as a green plant near the center of the cleared soil circle. A dead sapling may show a cleared soil circle but no green plant at the center, or just bare soil.

Task: Classify whether the sapling is ALIVE or DEAD in this crop.

Return ONLY valid JSON:
{
"status": "ALIVE" | "DEAD" | "UNCERTAIN",
"confidence": 0-100,
"evidence": ["short visual cues"]
}

Rules:
* If you cannot confidently decide, return UNCERTAIN with confidence <= 55.
* No extra text outside JSON."""
