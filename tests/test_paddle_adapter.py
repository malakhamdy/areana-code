import numpy as np

from egyptian_id_ocr.ocr.paddle_engine import _parse_prediction


class FakeResult:
    @property
    def json(self):
        return {
            "res": {
                "rec_texts": ["محمد", "٣٠٠"],
                "rec_scores": np.array([0.92, 0.88]),
                "rec_polys": np.array(
                    [
                        [[50, 10], [120, 10], [120, 35], [50, 35]],
                        [[10, 50], [80, 50], [80, 75], [10, 75]],
                    ]
                ),
            }
        }


def test_parse_paddle_3_payload_with_numpy_arrays():
    spans = _parse_prediction(FakeResult(), 0.15)
    assert [span.text for span in spans] == ["محمد", "٣٠٠"]
    assert spans[0].bbox[0] == [50.0, 10.0]
