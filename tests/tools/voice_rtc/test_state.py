from tools.voice_rtc.state import TurnState, Event


def test_starts_listening():
    s = TurnState()
    assert s.state == "LISTENING"
    assert s.cancel_requested is False


def test_listening_to_thinking_on_user_final():
    s = TurnState()
    s.handle(Event.USER_FINAL)
    assert s.state == "THINKING"


def test_thinking_to_speaking_on_first_audio():
    s = TurnState()
    s.handle(Event.USER_FINAL)
    s.handle(Event.TTS_FIRST_AUDIO)
    assert s.state == "SPEAKING"


def test_speaking_back_to_listening_on_done():
    s = TurnState()
    s.handle(Event.USER_FINAL)
    s.handle(Event.TTS_FIRST_AUDIO)
    s.handle(Event.TTS_DONE)
    assert s.state == "LISTENING"


def test_barge_in_during_speaking():
    s = TurnState()
    s.handle(Event.USER_FINAL)
    s.handle(Event.TTS_FIRST_AUDIO)
    s.handle(Event.VAD_SPEECH_START)
    assert s.state == "INTERRUPTED"
    assert s.cancel_requested is True


def test_barge_in_during_thinking():
    s = TurnState()
    s.handle(Event.USER_FINAL)
    s.handle(Event.VAD_SPEECH_START)
    assert s.state == "INTERRUPTED"
    assert s.cancel_requested is True


def test_cancel_done_returns_to_listening():
    s = TurnState()
    s.handle(Event.USER_FINAL)
    s.handle(Event.TTS_FIRST_AUDIO)
    s.handle(Event.VAD_SPEECH_START)
    s.handle(Event.CANCEL_DONE)
    assert s.state == "LISTENING"
    assert s.cancel_requested is False


def test_no_op_events_in_listening():
    s = TurnState()
    s.handle(Event.VAD_SPEECH_START)
    assert s.state == "LISTENING"
    assert s.cancel_requested is False


def test_unexpected_event_does_not_crash():
    s = TurnState()
    s.handle(Event.TTS_DONE)
    assert s.state == "LISTENING"
