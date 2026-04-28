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


# ---------------------------------------------------------------------------
# Generalized suppression — the chunker must defend against ANY known
# reasoning / tool-call shape that would otherwise stream into TTS as
# audible gibberish.
# ---------------------------------------------------------------------------


def test_strips_thinking_long_form():
    """Anthropic-style <thinking>...</thinking> must be stripped too."""
    c = ClauseChunker()
    out = c.feed("<thinking>internal monologue</thinking>The answer is yes.")
    assert out == ["The answer is yes."]


def test_strips_analysis_block():
    c = ClauseChunker()
    out = c.feed("<analysis>some analysis</analysis>Hello there.")
    assert out == ["Hello there."]


def test_strips_reflection_block():
    c = ClauseChunker()
    out = c.feed("<reflection>thoughts</reflection>I'm here.")
    assert out == ["I'm here."]


def test_strips_tool_call_markup():
    """Critical: when the model emits tool-call XML mid-response (the user
    hit this regression on Pravah), it must NOT stream into TTS."""
    c = ClauseChunker()
    out = c.feed("Let me check.")
    out += c.feed("<tool_call>skills_tool<arg_key>action</arg_key>")
    out += c.feed("<arg_value>view</arg_value></tool_call>")
    out += c.feed(" One moment.")
    # The tool_call block is fully suppressed.
    assert out == ["Let me check.", "One moment."]


def test_strips_tool_call_split_across_feeds():
    c = ClauseChunker()
    # Open tag straddles a feed boundary.
    out = c.feed("Reading. <tool_")
    out += c.feed("call>noise</tool_call>")
    out += c.feed(" Done.")
    assert out == ["Reading.", "Done."]


def test_pure_tool_call_response_emits_nothing():
    """Pravah was observed to return ONLY tool_call markup with no spoken
    text. Make sure the chunker swallows that completely instead of
    leaking the angle brackets / arg values to TTS."""
    c = ClauseChunker()
    out = c.feed(
        "<tool_call>skills_tool"
        "<arg_key>action</arg_key>"
        "<arg_value>view</arg_value>"
        "<arg_key>skill</arg_key>"
        "<arg_value>order_status</arg_value>"
        "</tool_call>"
    )
    out += c.flush()
    assert out == []


def test_unterminated_tool_call_drops_buffer_on_flush():
    c = ClauseChunker()
    c.feed("<tool_call>oops never closed")
    assert c.flush() == []


def test_literal_less_than_outside_tags_passes_through():
    """A bare '<' that doesn't lead into a known tag must reach output —
    we cannot blanket-suppress angle brackets."""
    c = ClauseChunker()
    out = c.feed("Five < six.")
    assert out == ["Five < six."]


def test_custom_suppress_set_replaces_default():
    """If the caller passes ``suppress_tags=()``, NOTHING is stripped."""
    c = ClauseChunker(suppress_tags=())
    out = c.feed("<think>kept verbatim.</think>")
    out += c.flush()
    assert out == ["<think>kept verbatim.", "</think>"]
