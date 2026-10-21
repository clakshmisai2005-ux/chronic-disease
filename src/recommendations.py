"""General, non-diagnostic health information for educational use."""


def get_recommendations(risk_class=None):
    guidance = [
        "Discuss personal risk factors and appropriate screening with a qualified healthcare professional.",
        "Regular movement, balanced meals, adequate sleep, and avoiding tobacco support general health.",
        "Do not start, stop, or change medication based on this estimate.",
    ]
    if risk_class == 1:
        guidance.insert(0, "Consider arranging a conversation with a healthcare professional to review your risk and screening options.")
    return guidance
