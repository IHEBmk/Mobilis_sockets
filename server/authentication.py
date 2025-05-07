# Replace your authentication.py with this:

from rest_framework_simplejwt.authentication import JWTAuthentication
from django.conf import settings
from rest_framework import exceptions
from rest_framework_simplejwt.tokens import RefreshToken

class CustomJWTAuthentication(JWTAuthentication):
    """
    Custom JWT Authentication that works with the existing User model
    """
    def get_user(self, validated_token):
        """
        Override to use your custom User model instead of Django's default
        """
        try:
            user_id = validated_token.get('user_id')
        except KeyError:
            raise exceptions.AuthenticationFailed('Token contained no recognizable user identification')

        from .models import User
        try:
            user = User.objects.get(id=user_id)
            # Add the is_authenticated property if it's not already in the model
            if not hasattr(user, 'is_authenticated'):
                user.is_authenticated = True
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed('User not found')

        return user
        
# Custom token generation function
def get_tokens_for_user(user):
    """
    Generate JWT tokens for a user
    """
    refresh = RefreshToken()
    
    # Add custom claims
    refresh['user_id'] = str(user.id)
    refresh['role'] = user.role
    refresh['first_name'] = user.first_name
    refresh['last_name'] = user.last_name
    
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }