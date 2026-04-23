from app.models.user import User
from app.schemas.user import UserCreate, UserResponse


_users: list[User] = []
_next_user_id = 1


def list_users() -> list[UserResponse]:
    return [UserResponse.model_validate(user) for user in _users]


def create_user(payload: UserCreate) -> UserResponse:
    global _next_user_id

    user = User(id=_next_user_id, name=payload.name, email=str(payload.email))
    _users.append(user)
    _next_user_id += 1
    return UserResponse.model_validate(user)
