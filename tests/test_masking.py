import pytest

from mono_agent.masking import mask_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("card 4441114455556789 ok", "card ****6789 ok"),
        ("card 4441 1144 5555 6789", "card ****6789"),
        ("card 4441-1144-5555-6789", "card ****6789"),
        ("IBAN UA213223130000026007233566001", "IBAN UA****6001"),
        ("ua21 3223 1300 0002 6007 2335 6600 1", "UA****6001"),
        ("already masked 537541******4821", "already masked 537541******4821"),
        ("amount 1500.00 UAH, order 123456", "amount 1500.00 UAH, order 123456"),
        (None, None),
    ],
)
def test_mask_text(raw, expected):
    assert mask_text(raw) == expected
