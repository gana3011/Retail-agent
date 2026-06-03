TEST_QUESTIONS = [
    # Glossary / Terms
    "What is a SKU and why is it important?",
    "What does UPC stand for?",
    "What is GMROI?",
    "What is the difference between ATP and safety stock?",
    "What is a planogram?",

    # Scenarios
    "What happens when a promo is not applied at POS?",
    "What went wrong when a customer's barcode wouldn't scan?",
    "Why did a customer not receive loyalty points after purchase?",
    "What happened when a duplicate order was created?",
    "Why was the wrong item delivered for an online order?",

    # Process Flows
    "What are the steps in the procurement process?",
    "How does inventory management work in retail?",
    "What is the order fulfillment process?",
    "How does the return management process work?",
    "What happens during the Sales and POS process?",

    # FAQs
    "What is omnichannel retail?",
    "What are the major retail formats?",
    "What is the order lifecycle in retail?",
    "What is reverse logistics?",
    "What is the difference between sell-in and sell-out?",

    # Training content
    "What is the retail value chain?",
    "What are the key retail KPIs to track?",
    "What is an omnichannel retail strategy?",
    "How is inventory turnover calculated?",
    "What is the role of POS system in retail?",
]

# Optional: mapping of questions to expected key terms for automated evaluation
EXPECTED_KEYWORDS: dict[str, list[str]] = {
    "What is a SKU and why is it important?": ["stock keeping unit", "product", "inventory"],
    "What does UPC stand for?": ["universal product code"],
    "What is the difference between ATP and safety stock?": ["available to promise", "safety stock"],
    "What is a planogram?": ["planogram", "shelf", "display"],
    "What is omnichannel retail?": ["omnichannel", "channel", "seamless"],
    "What is the order lifecycle in retail?": ["order", "lifecycle", "fulfillment"],
    "What is reverse logistics?": ["reverse", "logistics", "return"],
    "How is inventory turnover calculated?": ["inventory", "turnover", "ratio", "cogs"],
    "What is the role of POS system in retail?": ["pos", "point of sale", "transaction"],
}


def get_test_set() -> list[str]:
    return TEST_QUESTIONS


def evaluate_answer(question: str, answer: str) -> dict:
    keywords = EXPECTED_KEYWORDS.get(question, [])
    answer_lower = answer.lower()
    matched = [kw for kw in keywords if kw in answer_lower]
    return {
        "question": question,
        "keyword_hits": matched,
        "keyword_misses": [kw for kw in keywords if kw not in answer_lower],
        "precision": len(matched) / len(keywords) if keywords else None,
    }
