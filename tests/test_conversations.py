from tests.conftest import client, test_user


class TestConversations:
    """Test conversation endpoints."""
    
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
    
    def test_create_conversation(self, test_user):
        """Test conversation creation."""
        headers = self.get_auth_header(test_user)
        
        response = client.post(
            "/api/v1/conversations/",
            json={"title": "Test Conversation"},
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Conversation"
        assert "id" in data
        assert "user_id" in data
    
    def test_list_conversations(self, test_user):
        """Test listing conversations."""
        headers = self.get_auth_header(test_user)
        
        # Create conversation
        client.post(
            "/api/v1/conversations/",
            json={"title": "Test Conversation"},
            headers=headers
        )
        
        # List conversations
        response = client.get(
            "/api/v1/conversations/",
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_conversation(self, test_user):
        """Test getting a conversation."""
        headers = self.get_auth_header(test_user)
        
        # Create conversation
        create_response = client.post(
            "/api/v1/conversations/",
            json={"title": "Test Conversation"},
            headers=headers
        )
        conv_id = create_response.json()["id"]
        
        # Get conversation
        response = client.get(
            f"/api/v1/conversations/{conv_id}",
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == conv_id
        assert "messages" in data
    
    def test_delete_conversation(self, test_user):
        """Test conversation deletion."""
        headers = self.get_auth_header(test_user)
        
        # Create conversation
        create_response = client.post(
            "/api/v1/conversations/",
            json={"title": "Test Conversation"},
            headers=headers
        )
        conv_id = create_response.json()["id"]
        
        # Delete conversation
        response = client.delete(
            f"/api/v1/conversations/{conv_id}",
            headers=headers
        )
        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()
