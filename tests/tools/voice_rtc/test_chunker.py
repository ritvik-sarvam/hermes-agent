from tools.voice_rtc.chunker import ClauseChunker


def test_emits_on_period():
    c = ClauseChunker()
    assert c.feed("Hello there.") == ["Hello there."]


def test_buffers_until_clause():
    c = ClauseChunker()
    assert c.feed("Hello") == []
    assert c.feed(" there") == []
    assert c.feed(".") == ["Hello there."]


def test_emits_on_long_comma():
    c = ClauseChunker(min_chars_for_comma=10)
    assert c.feed("This is a fairly long clause, then more") == [
        "This is a fairly long clause,"
    ]


def test_flush_emits_remainder():
    c = ClauseChunker()
    c.feed("trailing")
    assert c.flush() == ["trailing"]


def test_emits_at_max_tokens():
    c = ClauseChunker(max_tokens=5)
    out = c.feed("one two three four five six")
    assert len(out) >= 1


def test_strips_think_block_inline():
    c = ClauseChunker()
    out = c.feed("<think>")
    out += c.feed("Some reasoning here. More reasoning.")
    out += c.feed("</think>")
    out += c.feed("Hello there.")
    assert out == ["Hello there."]


def test_strips_think_block_split_across_feeds():
    c = ClauseChunker()
    out = c.feed("<thi")
    out += c.feed("nk>internal monolog")
    out += c.feed("ue.</thi")
    out += c.feed("nk>The answer is yes.")
    assert out == ["The answer is yes."]


def test_handles_no_think_block():
    c = ClauseChunker()
    assert c.feed("Hi. How are you?") == ["Hi.", "How are you?"]


def test_flush_inside_think_block_emits_nothing():
    c = ClauseChunker()
    c.feed("<think>still reasoning, never closed")
    assert c.flush() == []
