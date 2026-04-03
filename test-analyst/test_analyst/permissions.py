from __future__ import annotations

from .configuration import ConfigManager
from .models import UserContext


class PermissionManager:
    """Very lightweight local role-based access control."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self.config_manager = config_manager

    def authenticate(self, username: str, password: str) -> UserContext | None:
        config = self.config_manager.permissions()
        for user in config.get("users", []):
            if user.get("username") == username and user.get("password") == password:
                return UserContext(username=username, role=user.get("role", "viewer"))
        return None

    def can(self, user: UserContext, action: str) -> bool:
        config = self.config_manager.permissions()
        allowed_actions = config.get("roles", {}).get(user.role, [])
        return action in allowed_actions

    def list_users(self) -> list[dict]:
        return self.config_manager.permissions().get("users", [])
