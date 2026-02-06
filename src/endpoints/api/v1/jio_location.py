import json
from pathlib import Path
from fastapi import APIRouter, Depends, Form, HTTPException, status
from src.core.utils.helpers import get_current_user



router = APIRouter(
    prefix="/api/jio-location",
    tags=["Hostel Management"]
)

LOCATION_FILE = Path("src/data/location_store.json")


import math

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in KM

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@router.post("/store")
def store_hostel_location(
    latitude: str = Form(...),
    longitude: str = Form(...),
    current_user=Depends(get_current_user),
):
    try:
        LOCATION_FILE.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "latitude": latitude,
            "longitude": longitude
        }

        with open(LOCATION_FILE, "w") as f:
            json.dump(data, f, indent=4)

        return {
            "status": True,
            "message": "Location stored successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/check")
def check_location_alert(
    current_lat: str = Form(...),
    current_long: str = Form(...),
    radius: int = Form(50),  # in KM
):
    try:
        if not LOCATION_FILE.exists():
            raise HTTPException(
                status_code=400,
                detail="school location not configured"
            )

        with open(LOCATION_FILE, "r") as f:
            stored_location = json.load(f)

        hostel_lat = float(stored_location["latitude"])
        hostel_long = float(stored_location["longitude"])

        distance = calculate_distance(
            float(current_lat),
            float(current_long),
            hostel_lat,
            hostel_long
        )

        if distance <= radius:
            return {
                "status": True,
                "message": "You are inside the allowed area",
                "distance_km": round(distance, 2)
            }
        else:
            return {
                "status": False,
                "message": "You are outside the allowed area",
                "distance_km": round(distance, 2)
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )



