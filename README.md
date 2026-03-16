# Diderot Validation Analysis: Beyond the Transaction

### **Subtitle:** Using Sentiment Mining on Social Media to Identify Psycholinguistic Markers of the Diderot Effect in Consumer Spending Cascades.

### Project Overview
This research investigates the Diderot Effect—a social phenomenon where obtaining a new possession creates a spiral of consumption—through the lens of Natural Language Processing (NLP). While previous studies have validated these purchase sequences using transactional data, this project focuses on the unstructured text of social media to identify the emotional and linguistic "early-warning signs" of financial risk.

### Research Question
Does the shift from functional to identity-signaling language in social media discourse serve as a predictive marker for a consumer entering a financially destabilizing Diderot spiral?

### The Three Building Blocks
- **Idea:** Psycholinguistic Early-Warning Signals
Move beyond post-facto sequence mining to predictive linguistics. The core hypothesis is that a "Diderot Spiral" is preceded by a measurable shift in Semantic Proximity: a consumer's language moves away from functional utility (e.g., "specs," "battery") and gravitates toward identity-cohesion (e.g., "aesthetic," "matching," "vibe"). We aim to quantify this "hedonic buildup" before it results in financial distress.
-  **Data:** Strategic Dual-Corpus Comparison (Reddit)
    1. The "Spiral" Corpus: 10,000+ posts from high-entry-barrier hobbyist communities (e.g., r/MechanicalKeyboards, r/Audiophile, r/HomeDecor). These act as the "observation lab" for identity-driven consumption.
    2. The "Impact" Corpus: 10,000+ posts from financial recovery communities (e.g., r/ShoppingAddiction, r/Debt, r/PersonalFinance). This provides the "ground truth" for the negative financial consequences of the effect.

- **Tools:** Python NLP & Predictive Modeling
  The analysis will move through a three-stage Python pipeline:
1. **Sentiment Dynamics (VADER & TextBlob):** To measure the "Emotional Arc"—tracking the high-arousal positive sentiment during the "spiral" phase versus the high-arousal negative sentiment (anxiety/regret) in the "impact" phase.

2. **Thematic Foundation Modeling (Gensim LDA):** To automatically cluster topics and detect the moment a user’s discourse shifts from "Need-based" to "Set-completion" themes.

3. **Vector Space Analysis (SpaCy / Word2Vec):** To calculate the Cosine Similarity between anchor products (NER-identified) and "Luxury/Aesthetic" word clusters, creating a numeric "Risk Score" for each user.


