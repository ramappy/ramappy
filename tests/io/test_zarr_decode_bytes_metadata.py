from ramappy.io.zarr.common import decode_dict


def test_decode_dict_decodes_nested_bytes_values_and_keys():
    payload = {
        b"level1": {
            b"name": b"Instrument A",
            "list": [b"x", {b"inner": b"y"}],
        }
    }

    decoded = decode_dict(payload)

    assert "level1" in decoded
    assert decoded["level1"]["name"] == "Instrument A"
    assert decoded["level1"]["list"][0] == "x"
    assert decoded["level1"]["list"][1]["inner"] == "y"
