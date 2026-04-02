from tests.conftest import client, test_user, test_document


class TestAuthentication:
    """Test authentication endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_register_user(self, test_user):
        """Test user registration."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": test_user["username"],
                "email": test_user["email"],
                "password": test_user["password"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user["username"]
        assert data["email"] == test_user["email"]
        assert "id" in data
        assert "is_active" in data
    
    def test_register_duplicate_user(self, test_user):
        """Test registering duplicate user."""
        # Register first user
        client.post(
            "/api/v1/auth/register",
            json={
                "username": test_user["username"],
                "email": test_user["email"],
                "password": test_user["password"]
            }
        )
        
        # Try to register same user
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": test_user["username"],
                "email": test_user["email"],
                "password": test_user["password"]
            }
        )
        assert response.status_code == 400
    
    def test_login_user(self, test_user):
        """Test user login."""
        # Register user first
        client.post(
            "/api/v1/auth/register",
            json={
                "username": test_user["username"],
                "email": test_user["email"],
                "password": test_user["password"]
            }
        )
        
        # Try to login
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": test_user["username"],
                "password": test_user["password"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_login_invalid_credentials(self, test_user):
        """Test login with invalid credentials."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "nonexistent",
                "password": "wrongpassword"
            }
        )
        assert response.status_code == 401
