#!/usr/bin/env python
"""
Initialize the database with sample data for testing and development.
"""

import os
import sys
from datetime import datetime
from sqlalchemy.orm import Session

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.db.database import SessionLocal, init_db
from app.db.models import User, Document
from app.core.security import get_password_hash
from app.services.document_service import DocumentService
from app.schemas import DocumentCreate


def seed_database():
    """Seed database with sample data."""
    
    # Initialize database
    print("Initializing database...")
    init_db()
    print("✓ Database initialized")
    
    db = SessionLocal()
    
    try:
        # Check if demo user already exists
        existing_user = db.query(User).filter(User.username == "demo").first()
        if existing_user:
            print("Demo user already exists, skipping creation")
            return
        
        # Create demo user
        print("Creating demo user...")
        demo_user = User(
            username="demo",
            email="demo@example.com",
            hashed_password=get_password_hash("demo1234"),
            is_active=True
        )
        db.add(demo_user)
        db.commit()
        db.refresh(demo_user)
        print(f"✓ Demo user created (ID: {demo_user.id})")
        
        # Create sample documents
        print("Creating sample documents...")
        document_service = DocumentService()
        
        sample_docs = [
            DocumentCreate(
                title="Password Reset Guide",
                content="""To reset your password:
1. Go to the login page
2. Click on "Forgot Password" link below the login button
3. Enter the email address associated with your account
4. Check your email for a password reset link
5. Click the link and follow the prompts to set a new password
6. You can now log in with your new password

Note: Password reset links expire in 24 hours for security reasons.""",
                source="help_documentation",
                metadata_json='{"category": "account", "priority": "high"}'
            ),
            DocumentCreate(
                title="Account Security Best Practices",
                content="""Keep your account secure by following these best practices:

1. Use Strong Passwords
   - At least 12 characters
   - Mix of uppercase, lowercase, numbers, and special characters
   - Don't use dictionary words or personal information

2. Two-Factor Authentication
   - Enable 2FA in account settings
   - Use authenticator apps like Google Authenticator or Authy
   - Backup codes should be stored safely

3. Session Management
   - Regularly log out from unused devices
   - Review active sessions in account settings
   - Log out after using shared computers

4. Email Security
   - Keep your recovery email updated
   - Enable email notifications for account activity
   - Verify email addresses for recovery options""",
                source="help_documentation",
                metadata_json='{"category": "security", "priority": "high"}'
            ),
            DocumentCreate(
                title="Billing and Subscription Guide",
                content="""Understanding Your Subscription:

Free Plan:
- 100 API calls per day
- Basic support
- Community access

Pro Plan ($9.99/month):
- 10,000 API calls per day
- Priority email support
- Beta feature access
- Advanced analytics

Enterprise Plan (Custom pricing):
- Unlimited API calls
- 24/7 phone support
- Dedicated account manager
- Custom integrations

Managing Your Subscription:
1. Go to Settings > Billing
2. View current plan and next billing date
3. Change plan or cancel anytime
4. Payments processed on the 1st of each month
5. Refunds issued within 30 days of cancellation""",
                source="help_documentation",
                metadata_json='{"category": "billing", "priority": "medium"}'
            ),
            DocumentCreate(
                title="Getting Started Guide",
                content="""Welcome to our platform! Here's how to get started:

Initial Setup (5 minutes):
1. Create an account with email and password
2. Verify your email address
3. Complete your profile
4. Choose a subscription plan

First Steps:
1. Navigate to the dashboard
2. Check out the quick start video
3. Read the API documentation
4. Create your first API key

Best Practices:
- Store API keys securely
- Use environment variables for credentials
- Test in sandbox mode first
- Monitor your API usage
- Keep your documentation handy

Need Help?
- Check out the FAQ section
- Browse the knowledge base
- Contact support: support@example.com
- Join our community forum""",
                source="help_documentation",
                metadata_json='{"category": "getting_started", "priority": "high"}'
            ),
            DocumentCreate(
                title="API Rate Limiting",
                content="""Understanding Rate Limits:

Rate limit rules are applied per plan:

Free Plan:
- 100 requests per day
- 10 requests per minute
- Burst limit: 5 requests per second

Pro Plan:
- 10,000 requests per day
- 100 requests per minute
- Burst limit: 50 requests per second

Enterprise Plan:
- Unlimited (for most use cases)
- Custom limits negotiable
- Burst protection available

What Happens When You Hit Limits?
- Requests return HTTP 429 (Too Many Requests)
- Response includes Retry-After header
- Try again after specified time

How to Increase Limits?
- Upgrade your subscription plan
- Contact support for enterprise requests
- Optimize your API usage patterns
- Use caching when applicable""",
                source="help_documentation",
                metadata_json='{"category": "api", "priority": "medium"}'
            ),
            DocumentCreate(
                title="Troubleshooting Common Issues",
                content="""Common Issues and Solutions:

Login Problems:
Issue: "Incorrect password" error
Solution: 
- Double-check caps lock
- Ensure email is correct
- Try password reset if you forgot it
- Clear browser cookies if stuck

Issue: "Email not verified"
Solution:
- Check spam/junk folder for verification email
- Click the verification link in the email
- Resend verification email from login page

API Errors:
Issue: 401 Unauthorized
Solution: Check your API key is valid and not expired

Issue: 403 Forbidden
Solution: Verify you have permission for that resource

Issue: 429 Too Many Requests
Solution: You've exceeded rate limit, wait before retrying

Issue: 500 Internal Server Error
Solution: Try again later, contact support if persists

Performance Issues:
Issue: Slow API responses
Solution:
- Check your connection speed
- Reduce request payload size
- Use pagination for large datasets
- Check server status page""",
                source="help_documentation",
                metadata_json='{"category": "troubleshooting", "priority": "high"}'
            )
        ]
        
        for doc in sample_docs:
            try:
                created_doc = document_service.add_document(db, doc)
                print(f"✓ Created document: {created_doc.title} (ID: {created_doc.id})")
            except Exception as e:
                print(f"✗ Error creating document {doc.title}: {str(e)}")
        
        print("\n✅ Database seeding completed successfully!")
        print("\nYou can now test the API with:")
        print("- Username: demo")
        print("- Password: demo1234")
        print("- Email: demo@example.com")
        
    except Exception as e:
        print(f"✗ Error during seeding: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
