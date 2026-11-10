import apeks
from apeks.app import move_camera
from apeks.config import CAM_SPEED, WORLD_SIZE

VIEW = (1280, 720)


def test_package_exports_main():
    assert apeks.main is apeks.app.main


def test_camera_pans_by_axis_and_speed():
    assert move_camera(100, 100, 1, -1, 0.1, VIEW) == (
        100 + CAM_SPEED * 0.1,
        100 - CAM_SPEED * 0.1,
    )


def test_camera_idle_without_input():
    assert move_camera(100, 100, 0, 0, 0.5, VIEW) == (100, 100)


def test_camera_clamped_to_world():
    assert move_camera(5, 5, -1, -1, 1.0, VIEW) == (0, 0)
    assert move_camera(WORLD_SIZE, WORLD_SIZE, 1, 1, 1.0, VIEW) == (
        WORLD_SIZE - VIEW[0],
        WORLD_SIZE - VIEW[1],
    )
