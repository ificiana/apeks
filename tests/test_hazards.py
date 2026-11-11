from apeks.hazards import HazardBook


def test_newer_observation_wins_and_confirmations_count() -> None:
    book = HazardBook()
    assert book.update("A>B", True, 10.0, "x") == "changed"
    assert book.update("A>B", True, 10.0, "y") is None
    assert book.entries["A>B"].sources == {"x", "y"}
    assert book.update("A>B", False, 5.0, "z") is None
    assert book.closed() == {"A>B"}
    assert book.update("A>B", True, 12.0, "x") == "refreshed"
    assert book.update("A>B", False, 20.0, "z") == "changed"
    assert book.closed() == set()
    assert book.entries["A>B"].sources == {"z"}


def test_fresh_entries_drop_old_open_notices_but_keep_closures() -> None:
    book = HazardBook()
    book.update("A>B", False, 10.0, "x")
    book.update("C>D", True, 10.0, "x")
    assert [lane for lane, _ in book.fresh(now=50.0, max_age=100.0)] == ["A>B", "C>D"]
    assert [lane for lane, _ in book.fresh(now=500.0, max_age=100.0)] == ["C>D"]


def test_observe_channel_updates_both_directions() -> None:
    from apeks.hazards import observe_channel, reverse

    book = HazardBook()
    assert reverse("A>B") == "B>A"
    assert observe_channel(book, "A>B", True, 1.0, "x") == "changed"
    assert book.closed() == {"A>B", "B>A"}
