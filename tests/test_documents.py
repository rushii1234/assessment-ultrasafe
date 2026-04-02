from tests.conftest import client, test_user, test_document
import json


class TestDocuments:
    """Test document endpoints."""
    
    @staticmethod
    def get_auth_header(test_user):
        """Helper to get authorization header."""
        # Register and login user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": test_user["username"],
                "email": test_user["email"],
                "password": test_user["password"]
            }
        )
        
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_create_document(self, test_user, test_document):
        """Test document creation."""
        headers = self.get_auth_header(test_user)
        
        response = client.post(
            "/api/v1/documents/",
            json=test_document,
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == test_document["title"]
        assert data["content"] == test_document["content"]
        assert "id" in data
    
    def test_list_documents(self, test_user, test_document):
        """Test listing documents."""
        headers = self.get_auth_header(test_user)
        
        # Create a document
        client.post(
            "/api/v1/documents/",
            json=test_document,
            headers=headers
        )
        
        # List documents
        response = client.get(
            "/api/v1/documents/",
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_get_document(self, test_user, test_document):
        """Test getting a specific document."""
        headers = self.get_auth_header(test_user)
        
        # Create a document
        create_response = client.post(
            "/api/v1/documents/",
            json=test_document,
            headers=headers
        )
        doc_id = create_response.json()["id"]
        
        # Get document
        response = client.get(
            f"/api/v1/documents/{doc_id}",
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == doc_id
        assert data["title"] == test_document["title"]
    
    def test_update_document(self, test_user, test_document):
        """Test document update."""
        headers = self.get_auth_header(test_user)
        
        # Create a document
        create_response = client.post(
            "/api/v1/documents/",
            json=test_document,
            headers=headers
        )
        doc_id = create_response.json()["id"]
        
        # Update document
        update_data = {"title": "Updated Title"}
        response = client.put(
            f"/api/v1/documents/{doc_id}",
            json=update_data,
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
    
    def test_delete_document(self, test_user, test_document):
        """Test document deletion."""
        headers = self.get_auth_header(test_user)
        
        # Create a document
        create_response = client.post(
            "/api/v1/documents/",
            json=test_document,
            headers=headers
        )
        doc_id = create_response.json()["id"]
        
        # Delete document
        response = client.delete(
            f"/api/v1/documents/{doc_id}",
            headers=headers
        )
        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()
