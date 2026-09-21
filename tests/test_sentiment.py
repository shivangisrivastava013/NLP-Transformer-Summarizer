from nlp_engine.sentiment import SentimentAnalyzer


def test_sentiment_empty():
    analyzer = SentimentAnalyzer(model_name="rule-based-fallback")
    res = analyzer.analyze("")
    assert res.label == "NEUTRAL"
    assert res.score == 1.0


def test_sentiment_positive():
    analyzer = SentimentAnalyzer(model_name="rule-based-fallback")
    text = "This breakthrough novel architecture achieved excellent and impressive high-performing results."
    res = analyzer.analyze(text)
    assert res.label == "POSITIVE"
    assert res.score > 0.5


def test_sentiment_negative():
    analyzer = SentimentAnalyzer(model_name="rule-based-fallback")
    text = "The system suffered severe error and failure due to inconsistent degraded performance loss."
    res = analyzer.analyze(text)
    assert res.label == "NEGATIVE"
    assert res.score > 0.5
