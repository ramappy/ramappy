import numpy as np
import pytest
from PIL import Image

from ramappy.core.collections import OrderedEntityMap
from ramappy.core.images2d import Image2D


def test_transaction_in_place_tracking_and_rollback():
    m = OrderedEntityMap({"a": {"values": np.array([1.0, np.nan])}, "b": [1]})
    with m.begin_transaction() as tx:
        pass
    assert tx.is_empty
    with m.begin_transaction() as tx:
        m["a"]["values"][0] = 7.0
        m["b"].append(2)
    assert set(tx.modified) == {"a", "b"}
    with pytest.raises(RuntimeError), m.begin_transaction():
        m["a"]["values"][0] = 12.0
        m["b"].append(3)
        raise RuntimeError("rollback")
    np.testing.assert_equal(m["a"]["values"], [7.0, np.nan])
    assert m["b"] == [1, 2]


def test_transaction_tracks_masked_and_raster_images():
    masked = np.ma.masked_array([[1.0, 2.0]], mask=[[False, True]])
    m = OrderedEntityMap(
        {
            "masked": Image2D(data=masked),
            "raster": Image2D(image=Image.new("RGB", (2, 1))),
        }
    )
    with m.begin_transaction() as tx:
        pass
    assert tx.is_empty
    with m.begin_transaction() as tx:
        m["masked"].data.mask[0, 0] = True
        m["raster"]._image.putpixel((0, 0), (255, 0, 0))
    assert set(tx.modified) == {"masked", "raster"}
