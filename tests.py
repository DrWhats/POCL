from main import app

def test_index():
    response = app.test_client().get('/')
    assert response.status_code == 200