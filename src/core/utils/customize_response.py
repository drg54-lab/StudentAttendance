from pydantic import BaseModel
from typing import Any, Optional, Generic, TypeVar, List

class AuthAPIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    user : Optional[Any] = None
    error: Optional[Any] = None

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[Any] = None

def success_response(message: str, data=None):
    return APIResponse(
        success=True,
        message=message,
        data=data,
        error=None
    )

def success_login_response(message: str, data=None, user=None):
    return AuthAPIResponse(
        success=True,
        message=message,
        data=data,
        user=user,
        error=None
    )

def error_response(message: str, code: int, details=None):
    return APIResponse(
        success=False,
        message=message,
        data=None,
        error={
            "code": code,
            "details": details
        }
    )

def login_error_response(message: str, code: int, details=None):
    return AuthAPIResponse(
        success=False,
        message=message,
        data=None,
        user=None,
        error={
            "code": code,
            "details": details
        }
    )

T = TypeVar("T")

class PaginationMeta(BaseModel):
    """
    Standard pagination metadata.
    """
    current_page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool

class PaginatedResponse(BaseModel, Generic[T]):
    """
    Generic paginated response structure.
    Use this when returning lists with pagination.
    """
    success: bool = True
    message: str
    data: List[T]                # the actual list of items
    pagination: PaginationMeta
    error: Optional[Any] = None

def paginated_success_response(
    message: str,
    items: List[T],
    current_page: int,
    page_size: int,
    total_items: int,
) -> PaginatedResponse[T]:
    """
    Helper to create a standardized paginated success response.

    Example usage:
        return paginated_success_response(
            message="Users retrieved successfully",
            items=user_list,
            current_page=page,
            page_size=limit,
            total_items=total_count
        )
    """
    total_pages = (total_items + page_size - 1) // page_size if page_size > 0 else 0

    pagination = PaginationMeta(
        current_page=current_page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        has_next=current_page < total_pages,
        has_previous=current_page > 1
    )

    return PaginatedResponse[T](
        success=True,
        message=message,
        data=items,
        pagination=pagination,
        error=None
    )
