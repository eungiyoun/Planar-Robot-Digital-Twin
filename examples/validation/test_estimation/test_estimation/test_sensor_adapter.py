import numpy as np
import pytest

from rse.system import Robot
from rse.sensor import GPSCompass, Gyro
from rse.estimation import SensorAdapter


def _make_robot() -> Robot:
    """Create a robot with a known state for deterministic testing."""
    robot = Robot(mass=2.0, moment_of_inertia=0.05, radius=0.2, track_width=0.15)
    robot.position = np.array([1.0, 2.0])
    # Set heading to pi/4 via orientation matrix
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    robot.orientation = np.array([[c, -s], [s, c]])
    robot.angular_momentum = 0.05 * 5.0  # omega_true = 5.0 rad/s
    return robot


# ── Protocol Conformance ──────────────────────────────────────────────


class TestProtocolConformance:
    """SensorAdapter must structurally satisfy SensorReadingProtocol."""

    def test_is_structural_subtype(self):
        # Check that the class defines the required property descriptors
        assert isinstance(SensorAdapter.y_gps, property)
        assert isinstance(SensorAdapter.y_compass, property)
        assert isinstance(SensorAdapter.y_gyro, property)

    def test_isinstance_check_with_runtime_checkable(self):
        """Verify adapter satisfies the protocol via runtime_checkable if available."""
        from typing import runtime_checkable, Protocol

        @runtime_checkable
        class _CheckableSensorReading(Protocol):
            @property
            def y_gps(self) -> np.ndarray: ...
            @property
            def y_compass(self) -> float: ...
            @property
            def y_gyro(self) -> float: ...

        adapter = SensorAdapter(GPSCompass(seed=0), Gyro(seed=0))
        assert isinstance(adapter, _CheckableSensorReading)


# ── Delegation ────────────────────────────────────────────────────────


class TestMeasureDelegation:
    """measure() must call both underlying sensors."""

    def test_measure_populates_all_readings(self):
        adapter = SensorAdapter(GPSCompass(seed=0), Gyro(seed=0))
        robot = _make_robot()
        adapter.measure(robot)
        # All three properties should now be accessible without error
        _ = adapter.y_gps
        _ = adapter.y_compass
        _ = adapter.y_gyro

    def test_measure_returns_none(self):
        adapter = SensorAdapter(GPSCompass(seed=0), Gyro(seed=0))
        result = adapter.measure(_make_robot())
        assert result is None

    def test_repeated_measure_updates_readings(self):
        adapter = SensorAdapter(GPSCompass(seed=0), Gyro(seed=0))
        robot = _make_robot()
        adapter.measure(robot)
        first_gps = adapter.y_gps.copy()
        adapter.measure(robot)
        second_gps = adapter.y_gps.copy()
        # With noise, two successive readings should differ
        assert not np.array_equal(first_gps, second_gps)


# ── Property Types and Shapes ─────────────────────────────────────────


class TestPropertyTypesAndShapes:
    """After measure(), properties must return correct types and shapes."""

    @pytest.fixture()
    def adapter(self):
        a = SensorAdapter(GPSCompass(seed=42), Gyro(seed=42))
        a.measure(_make_robot())
        return a

    def test_y_gps_is_ndarray_shape_2(self, adapter):
        gps = adapter.y_gps
        assert isinstance(gps, np.ndarray)
        assert gps.shape == (2,)

    def test_y_compass_is_float(self, adapter):
        assert isinstance(adapter.y_compass, (float, np.floating))

    def test_y_gyro_is_float(self, adapter):
        assert isinstance(adapter.y_gyro, (float, np.floating))

    def test_y_gps_near_true_position(self, adapter):
        """GPS should be within a few sigma of the true position [1, 2]."""
        gps = adapter.y_gps
        assert abs(gps[0] - 1.0) < 1.0  # within ~10 sigma_gps
        assert abs(gps[1] - 2.0) < 1.0

    def test_y_compass_near_true_heading(self, adapter):
        """Compass should be within a few sigma of pi/4."""
        assert abs(adapter.y_compass - np.pi / 4) < 3.0  # within ~6 sigma_compass

    def test_y_gyro_near_true_omega(self, adapter):
        """Gyro should be within a few sigma of 5.0 rad/s."""
        assert abs(adapter.y_gyro - 5.0) < 3.0  # within ~15 sigma_gyro


# ── RuntimeError Before measure() ─────────────────────────────────────


class TestAccessBeforeMeasure:
    """Accessing properties before measure() must raise RuntimeError."""

    def test_y_gps_raises(self):
        adapter = SensorAdapter(GPSCompass(), Gyro())
        with pytest.raises(RuntimeError, match="measure"):
            _ = adapter.y_gps

    def test_y_compass_raises(self):
        adapter = SensorAdapter(GPSCompass(), Gyro())
        with pytest.raises(RuntimeError, match="measure"):
            _ = adapter.y_compass

    def test_y_gyro_raises(self):
        adapter = SensorAdapter(GPSCompass(), Gyro())
        with pytest.raises(RuntimeError, match="measure"):
            _ = adapter.y_gyro
