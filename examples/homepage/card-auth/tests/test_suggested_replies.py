from bank import ON_FILE, Bank, Identity
from card_services import caller_script
from suggested_replies import parse_suggestions

JORDAN = ON_FILE[0]


def test_parses_the_models_json():
    raw = '{"suggestions": [{"label": "My name", "value": "It\'s Jordan Rivera."}]}'

    assert parse_suggestions(raw) == [
        {"label": "My name", "value": "It's Jordan Rivera."}
    ]


def test_tolerates_fences_and_prose_around_the_json():
    raw = 'Sure:\n```json\n{"suggestions": [{"label": "Yes", "value": "Yes, that\'s right."}]}\n```'

    assert parse_suggestions(raw) == [{"label": "Yes", "value": "Yes, that's right."}]


def test_drops_anything_that_is_not_a_labelled_line():
    raw = (
        '{"suggestions": [{"label": "", "value": "x"}, {"value": "no label"}, "text",'
        ' {"label": " Ok ", "value": " Okay. "}]}'
    )

    assert parse_suggestions(raw) == [{"label": "Ok", "value": "Okay."}]


def test_caps_the_list_and_survives_garbage():
    four = [{"label": str(i), "value": f"line {i}"} for i in range(4)]

    assert len(parse_suggestions(str(four).replace("'", '"'))) == 3
    assert parse_suggestions("I don't know") == []


async def test_script_gets_the_purchase_wrong_first_then_exact_then_wraps_up():
    bank = Bank(lookup_seconds=0)

    first = caller_script(bank)
    assert "Jordan Rivera" in first and "April 22, 1990" in first and "4417" in first
    assert "about forty dollars" in first

    await bank.authenticate(
        Identity(
            name=JORDAN.name,
            date_of_birth=JORDAN.date_of_birth,
            last_four=JORDAN.last_four,
            merchant=JORDAN.merchant,
            amount="about forty dollars",
        )
    )
    retry = caller_script(bank)
    assert "about forty dollars" not in retry
    assert "exactly as on file" in retry and "42.18" in retry

    await bank.authenticate(JORDAN)
    assert "goodbye" in caller_script(bank)
