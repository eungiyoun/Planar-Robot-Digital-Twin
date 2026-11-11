import numpy as np
import pytest

from rse.estimation import Estimator
from tests.test_estimation._mock_sensor import MockSensor


# ── Initialization ────────────────────────────────────────────────────


class TestInitialization:
    """First update() seeds the internal state from the sensor."""

    def test_first_update_seeds_position(self):
        sensor = MockSensor(y_gps=np.array([3.0, 4.0]))
        est = Estimator(sensor)
        est.update()
        np.testing.assert_array_equal(est.x_hat, [3.0, 4.0])

    def test_first_update_seeds_heading(self):
        sensor = MockSensor(y_compass=1.23)
        est = Estimator(sensor)
        est.update()
        assert est.theta_hat == pytest.approx(1.23)

    def test_first_update_seeds_rotation_matrix(self):
        sensor = MockSensor(y_compass=np.pi / 2)
        est = Estimator(sensor)
        est.update()
        Q = est.Q_hat
        expected = np.array([[0, -1], [1, 0]], dtype=float)
        np.testing.assert_allclose(Q, expected, atol=1e-12)


class TestAccessBeforeUpdate:
    """Accessing estimates before any update() must raise RuntimeError."""

    def test_x_hat_raises(self):
        est = Estimator(MockSensor())
        with pytest.raises(RuntimeError):
            _ = est.x_hat

    def test_theta_hat_raises(self):
        est = Estimator(MockSensor())
        with pytest.raises(RuntimeError):
            _ = est.theta_hat

    def test_Q_hat_raises(self):
        est = Estimator(MockSensor())
        with pytest.raises(RuntimeError):
            _ = est.Q_hat

    def test_pose_raises(self):
        est = Estimator(MockSensor())
        with pytest.raises(RuntimeError):
            _ = est.pose


# ── Low-Pass Filter (Position) ───────────────────────────────────────


class TestLowPassFilter:
    """Position low-pass filter: x_hat_k = (1-a)*x_hat_{k-1} + a*y_gps_k."""

    def test_constant_gps_converges(self):
        """Constant GPS reading → estimate converges to that value."""
        true_pos = np.array([5.0, -3.0])
        sensor = MockSensor(y_gps=true_pos)
        est = Estimator(sensor, alpha_x=0.05)
        for _ in range(2000):
            est.update()
        np.testing.assert_allclose(est.x_hat, true_pos, atol=1e-6)

    def test_alpha_near_one_passes_raw_gps(self):
        """alpha_x close to 1 → estimate tracks GPS immediately."""
        sensor = MockSensor(y_gps=np.array([1.0, 1.0]))
        est = Estimator(sensor, alpha_x=0.99)
        est.update()  # seed
        sensor.y_gps = np.array([10.0, 10.0])
        est.update()
        # After one step with alpha=0.99: 0.01*1 + 0.99*10 = 10.0 - 0.09 ≈ 9.91
        np.testing.assert_allclose(est.x_hat, [9.91, 9.91], atol=0.01)

    def test_alpha_near_zero_heavy_smoothing(self):
        """alpha_x close to 0 → estimate barely moves from seed."""
        sensor = MockSensor(y_gps=np.array([1.0, 1.0]))
        est = Estimator(sensor, alpha_x=0.001)
        est.update()  # seed at [1, 1]
        sensor.y_gps = np.array([100.0, 100.0])
        est.update()
        # Should barely move: (1-0.001)*1 + 0.001*100 = 0.999 + 0.1 = 1.099
        np.testing.assert_allclose(est.x_hat, [1.099, 1.099], atol=0.01)

    def test_filtered_variance_less_than_raw(self):
        """Filtered GPS has lower variance than raw noisy GPS."""
        rng = np.random.default_rng(42)
        true_pos = np.array([2.0, 3.0])
        sensor = MockSensor(y_gps=true_pos.copy())
        est = Estimator(sensor, alpha_x=0.02)

        raw_errors = []
        filtered_errors = []
        for _ in range(5000):
            noisy = true_pos + rng.normal(0, 0.1, size=2)
            sensor.y_gps = noisy
            est.update()
            raw_errors.append(np.linalg.norm(noisy - true_pos))
            filtered_errors.append(np.linalg.norm(est.x_hat - true_pos))

        # Skip transient (first 100 steps)
        raw_var = np.var(raw_errors[100:])
        filt_var = np.var(filtered_errors[100:])
        assert filt_var < raw_var


# ── Complementary Filter (Orientation) ───────────────────────────────


class TestComplementaryFilter:
    """Heading complementary filter tests."""

    def test_static_heading_converges_to_compass(self):
        """Zero gyro, constant compass → heading converges to compass."""
        sensor = MockSensor(y_compass=0.5, y_gyro=0.0)
        est = Estimator(sensor, alpha_theta=0.98, dt=0.001)
        for _ in range(50_000):
            est.update()
        assert est.theta_hat == pytest.approx(0.5, abs=1e-3)

    def test_gyro_dominates_short_term(self):
        """High alpha_theta → gyro integration dominates over a few steps."""
        sensor = MockSensor(y_compass=0.0, y_gyro=10.0)
        est = Estimator(sensor, alpha_theta=0.99, dt=0.001)
        est.update()  # seed at compass = 0.0

        # 10 steps: gyro predicts theta ≈ 10 * 0.001 * 10 = 0.1 rad
        for _ in range(10):
            est.update()

        # Compass says 0.0, gyro pushes positive — estimate should be positive
        assert est.theta_hat > 0.05

    def test_compass_corrects_drift_long_term(self):
        """Over many steps, compass pulls the estimate toward its reading."""
        sensor = MockSensor(y_compass=1.0, y_gyro=0.0)
        est = Estimator(sensor, alpha_theta=0.98, dt=0.001)
        est.update()  # seed at compass = 1.0

        # Now compass reads 0.5, gyro still 0 → estimate drifts toward 0.5
        sensor.y_compass = 0.5
        for _ in range(100_000):
            est.update()
        assert est.theta_hat == pytest.approx(0.5, abs=1e-2)


# ── Rotation Matrix Properties ────────────────────────────────────────


class TestRotationMatrix:
    """Q_hat must be a valid SO(2) matrix."""

    def test_orthogonality(self):
        sensor = MockSensor(y_compass=0.7)
        est = Estimator(sensor)
        est.update()
        Q = est.Q_hat
        np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-12)

    def test_determinant_positive_one(self):
        sensor = MockSensor(y_compass=1.3)
        est = Estimator(sensor)
        est.update()
        assert np.linalg.det(est.Q_hat) == pytest.approx(1.0, abs=1e-12)

    def test_identity_at_zero(self):
        sensor = MockSensor(y_compass=0.0)
        est = Estimator(sensor)
        est.update()
        np.testing.assert_allclose(est.Q_hat, np.eye(2), atol=1e-12)

    def test_correct_at_pi_over_2(self):
        sensor = MockSensor(y_compass=np.pi / 2)
        est = Estimator(sensor)
        est.update()
        expected = np.array([[0.0, -1.0], [1.0, 0.0]])
        np.testing.assert_allclose(est.Q_hat, expected, atol=1e-12)


# ── Gain Utilities ─────────────────────────────────────────────────────


class TestGainUtilities:
    """Static helper methods for gain / frequency conversion."""

    def test_gain_from_cutoff_range(self):
        """Result must be in (0, 1) for reasonable inputs."""
        alpha = Estimator.gain_from_cutoff(omega_c=10.0, dt=0.001)
        assert 0 < alpha < 1

    def test_gain_from_cutoff_value(self):
        """alpha = 1 - exp(-omega_c * dt)."""
        alpha = Estimator.gain_from_cutoff(omega_c=50.0, dt=0.01)
        expected = 1.0 - np.exp(-50.0 * 0.01)
        assert alpha == pytest.approx(expected)

    def test_gain_increases_with_cutoff(self):
        a_low = Estimator.gain_from_cutoff(1.0, 0.001)
        a_high = Estimator.gain_from_cutoff(100.0, 0.001)
        assert a_high > a_low

    def test_crossover_frequency_formula(self):
        """omega_cross = -ln(alpha_theta) / dt."""
        omega = Estimator.crossover_frequency(0.98, 0.001)
        expected = -np.log(0.98) / 0.001
        assert omega == pytest.approx(expected)

    def test_crossover_frequency_invalid_alpha(self):
        with pytest.raises(ValueError):
            Estimator.crossover_frequency(0.0, 0.001)
        with pytest.raises(ValueError):
            Estimator.crossover_frequency(1.0, 0.001)


# ── Gain Validation ────────────────────────────────────────────────────


class TestGainValidation:
    """Setting gains outside (0, 1) must raise ValueError."""

    def test_alpha_x_zero_raises(self):
        with pytest.raises(ValueError):
            Estimator(MockSensor(), alpha_x=0.0)

    def test_alpha_x_one_raises(self):
        with pytest.raises(ValueError):
            Estimator(MockSensor(), alpha_x=1.0)

    def test_alpha_x_negative_raises(self):
        with pytest.raises(ValueError):
            Estimator(MockSensor(), alpha_x=-0.1)

    def test_alpha_theta_zero_raises(self):
        with pytest.raises(ValueError):
            Estimator(MockSensor(), alpha_theta=0.0)

    def test_alpha_theta_one_raises(self):
        with pytest.raises(ValueError):
            Estimator(MockSensor(), alpha_theta=1.0)

    def test_alpha_theta_above_one_raises(self):
        with pytest.raises(ValueError):
            Estimator(MockSensor(), alpha_theta=1.5)

    def test_setter_rejects_invalid_alpha_x(self):
        est = Estimator(MockSensor())
        with pytest.raises(ValueError):
            est.alpha_x = 0.0

    def test_setter_rejects_invalid_alpha_theta(self):
        est = Estimator(MockSensor())
        with pytest.raises(ValueError):
            est.alpha_theta = 1.0
