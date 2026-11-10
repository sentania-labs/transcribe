from app.assign import assign_speakers, speaker_at


def test_words_follow_turns():
    turns = [(0.0, 5.0, "A"), (5.0, 9.0, "B"), (9.0, 12.0, "A")]
    words = [(0.5, 1.0, " Hello"), (1.1, 1.5, " there."), (5.2, 5.6, " Hi"),
             (5.7, 6.0, " back."), (9.5, 10.0, " Bye.")]
    segs = assign_speakers(words, turns)
    assert [(s["speaker"], s["text"]) for s in segs] == [
        ("Speaker 1", "Hello there."), ("Speaker 2", "Hi back."), ("Speaker 1", "Bye.")]
    assert segs[0]["start"] == 0.5 and segs[0]["end"] == 1.5
    assert [s["id"] for s in segs] == [0, 1, 2]


def test_word_spanning_a_change_goes_to_bigger_overlap():
    turns = [(0.0, 5.0, "A"), (5.0, 9.0, "B")]
    starts = [t[0] for t in turns]
    assert speaker_at(4.9, 5.5, turns, starts) == "B"
    assert speaker_at(4.5, 5.1, turns, starts) == "A"


def test_word_in_a_gap_goes_to_nearest_turn():
    turns = [(0.0, 2.0, "A"), (10.0, 12.0, "B")]
    starts = [t[0] for t in turns]
    assert speaker_at(2.5, 2.8, turns, starts) == "A"
    assert speaker_at(9.0, 9.5, turns, starts) == "B"


def test_no_turns_still_returns_text():
    segs = assign_speakers([(0.0, 1.0, " Only"), (1.0, 2.0, " words.")], [])
    assert segs == [{"id": 0, "speaker": "Speaker 1", "start": 0.0, "end": 2.0, "text": "Only words."}]


def test_long_pause_and_max_length_split():
    turns = [(0.0, 200.0, "A")]
    words = [(0.0, 1.0, " a"), (10.0, 11.0, " b")]  # 9 s pause
    assert len(assign_speakers(words, turns)) == 2
    words = [(float(t), t + 0.9, " w") for t in range(0, 130)]  # continuous 130 s
    segs = assign_speakers(words, turns)
    assert len(segs) > 1 and all(s["end"] - s["start"] <= 60 for s in segs)
    assert {s["speaker"] for s in segs} == {"Speaker 1"}


def test_sentence_stays_with_its_speaker_across_a_late_turn_change():
    # Diarization switches to B at 3.0, but Whisper's last word of A's sentence
    # starts at 2.9: word by word it would go to B.
    turns = [(0.0, 3.0, "A"), (3.0, 8.0, "B")]
    words = [(0.0, 0.5, " Now,"), (0.6, 1.0, " if"), (1.1, 1.5, " I"), (1.6, 2.8, " have"),
             (2.9, 3.4, " to."), (3.6, 4.0, " Well,"), (4.1, 4.5, " no.")]
    segs = assign_speakers(words, turns)
    assert [(s["speaker"], s["text"]) for s in segs] == [
        ("Speaker 1", "Now, if I have to."), ("Speaker 2", "Well, no.")]


def test_interrupted_sentence_ends_at_the_dash():
    turns = [(0.0, 2.0, "A"), (2.0, 6.0, "B")]
    words = [(0.0, 0.5, " I"), (0.6, 1.0, " will"), (1.1, 1.8, " go"), (1.9, 2.4, " through—"),
             (2.5, 3.0, " Well,"), (3.1, 3.5, " but"), (3.6, 4.0, " no.")]
    segs = assign_speakers(words, turns)
    assert [(s["speaker"], s["text"]) for s in segs] == [
        ("Speaker 1", "I will go through—"), ("Speaker 2", "Well, but no.")]
