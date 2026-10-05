import pytest

from motionsense_app.errors import DomainError


def validate(content, required=None, known=None):
    from motionsense_app.recognition.csv_input import validate_csv

    return validate_csv(content, required or ["f001"], known or ["f001", "f002"])


def test_bom_reorders_features_and_separates_metadata():
    rows, labels = validate(
        b'\xef\xbb\xbff002,activity,subject_id,f001\r\n"0.2",1,2,0.1\r\n',
        ["f001", "f002"],
    )
    assert rows == [{"f001": 0.1, "f002": 0.2}]
    assert labels == [1]
    assert validate(b"f001\n0.1\n")[1] == [None]


@pytest.mark.parametrize("value", ["NaN", "inf", "-inf", "", "abc", "1e309", "1e300"])
def test_required_numeric_errors_include_file_row_and_column(value):
    with pytest.raises(DomainError) as caught:
        validate(f"f001\n0.1\n{value}\n".encode())
    assert caught.value.status == 422
    assert caught.value.details[0]["column"] == "f001"
    assert caught.value.details[0]["row"] == 3


@pytest.mark.parametrize("content,column,row", [
    (b"f001,f001\n1,2\n", "f001", 1),
    (b"f002\n1\n", "f001", 1),
    (b"f001,mystery\n1,2\n", "mystery", 1),
    (b"f001,\n1,2\n", "", 1),
    (b"f001,f002\n1\n", "f002", 2),
    (b"f001\n1,2\n", None, 2),
    (b"f001\n\n", "f001", 2),
    (b"f001\n", "f001", 2),
    (b"", "f001", 1),
    (b"\xef\xbb\xbf", "f001", 1),
    (b"\n", "f001", 1),
    (b'f001,f002\n0.1,"unterminated\n', "f002", 2),
    (b'f001,f002\n0.1,"0.2"junk\n', "f002", 2),
    (b'f001,f002\n"0.1\n",NaN\n', "f002", 2),
    (b"f001,f002\n0.1,NaN\n", "f002", 2),
])
def test_structural_and_extra_feature_errors_are_localized(content, column, row):
    with pytest.raises(DomainError) as caught:
        validate(content)
    assert caught.value.status == 422
    assert caught.value.details[0]["row"] == row
    assert caught.value.details[0]["column"] == column


@pytest.mark.parametrize("column,values", [
    ("activity", ["0", "7", "1.1", "1.0000000000000001", "", "NaN", "inf", "abc"]),
    ("subject_id", ["0", "-1", "2.1", "2.0000000000000001", "", "NaN", "inf", "abc"]),
])
def test_metadata_must_be_finite_integers_in_range(column, values):
    for value in values:
        with pytest.raises(DomainError) as caught:
            validate(f"f001,{column}\n0.1,{value}\n".encode())
        assert caught.value.status == 422
        assert caught.value.details[0]["row"] == 2
        assert caught.value.details[0]["column"] == column


def test_invalid_utf8_is_localized():
    with pytest.raises(DomainError) as caught:
        validate(b"f001,f002\n1,2\n3,\xff\n")
    assert caught.value.status == 422
    assert caught.value.details[0]["row"] == 3
    assert caught.value.details[0]["column"] == "f002"


def test_whole_file_rejected_and_error_details_capped_with_total_count():
    with pytest.raises(DomainError) as caught:
        validate(b"f001,f002\n0.1,0.2\n" + b"NaN,abc\n" * 30)
    assert caught.value.status == 422
    assert len(caught.value.details) == 50
    assert "60" in caught.value.message
    assert caught.value.details[0]["row"] == 3
    assert caught.value.details[-1]["row"] == 27


def test_row_limit_accepts_5000_and_rejects_5001():
    assert len(validate(b"f001\n" + b"1\n" * 5000)[0]) == 5000
    with pytest.raises(DomainError) as caught:
        validate(b"f001\n" + b"1\n" * 5001)
    assert caught.value.status == 413
    assert "5000" in caught.value.message
    assert caught.value.details[0]["row"] == 5002


def test_byte_limit_accepts_exactly_16_mib_and_rejects_one_more():
    limit = 16 * 1024 * 1024
    # Keep individual fields below csv.reader's field limit, at exactly the byte boundary.
    base = b"f001\n" + (b"1" + b" " * 3500 + b"\n") * 4790
    content = base + b"1" + b" " * (limit - len(base) - 2) + b"\n"
    assert len(content) == limit
    assert len(validate(content)[0]) == 4791
    with pytest.raises(DomainError) as caught:
        validate(content + b" ")
    assert caught.value.status == 413
    assert "16 MiB" in caught.value.message


def test_valid_long_numeric_field_has_no_undocumented_smaller_limit():
    assert validate(b"f001\n1" + b" " * 200000 + b"\n") == ([{"f001": 1.0}], [None])
