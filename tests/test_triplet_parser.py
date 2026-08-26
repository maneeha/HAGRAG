from hagrag.triplets import parse_triplet_response


def test_parser_accepts_fenced_json():
    response = '''```json
    {"entities":[{"name":"Metformin","type":"Drug","description":"Antidiabetic drug","attributes":{}}],
     "relationships":[{"source":"Metformin","target":"T2D","type":"TREATS","description":"used in management","attributes":{}}]}
    ```'''
    entities, relationships = parse_triplet_response(response, "doc-1")
    assert entities[0].name == "Metformin"
    assert relationships[0].target == "T2D"
    assert relationships[0].document_id == "doc-1"


def test_parser_returns_empty_for_invalid_output():
    assert parse_triplet_response("not json") == ([], [])
