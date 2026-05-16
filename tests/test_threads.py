# tests/test_threads.py
from xkb.extract.threads import assemble_thread


def _tweet(rest_id: str, handle: str, text: str, created: str) -> dict:
    return {
        "content": {
            "itemContent": {
                "tweet_results": {
                    "result": {
                        "__typename": "Tweet",
                        "rest_id": rest_id,
                        "core": {
                            "user_results": {
                                "result": {
                                    "legacy": {"screen_name": handle, "name": handle}
                                }
                            }
                        },
                        "legacy": {
                            "full_text": text,
                            "created_at": created,
                            "entities": {"urls": []},
                        },
                    }
                }
            }
        }
    }


def _tweet_detail(*tweets: dict) -> dict:
    return {
        "data": {
            "threaded_conversation_with_injections_v2": {
                "instructions": [
                    {"type": "TimelineAddEntries", "entries": list(tweets)}
                ]
            }
        }
    }


def test_assemble_thread_concatenates_author_tweets_in_order():
    response = _tweet_detail(
        _tweet("2", "alice", "second part", "Wed May 10 14:25:00 +0000 2026"),
        _tweet("1", "alice", "first part", "Wed May 10 14:23:00 +0000 2026"),
        _tweet("9", "bob", "a reply", "Wed May 10 14:30:00 +0000 2026"),
    )
    assert assemble_thread([response], "alice") == "first part\n\nsecond part"


def test_assemble_thread_empty_when_no_author_tweets():
    response = _tweet_detail(
        _tweet("9", "bob", "reply", "Wed May 10 14:30:00 +0000 2026")
    )
    assert assemble_thread([response], "alice") == ""
