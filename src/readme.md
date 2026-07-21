# Extended Kalman Filter (EKF) - Conceptual Flow

## Overview

A Kalman Filter estimates the hidden state of a system by combining:

* A prediction from the motion model.
* A correction from noisy sensor measurements.

The Extended Kalman Filter (EKF) follows the same idea but is used when the system or measurement models are nonlinear.

---

# 1. Initialization

The EKF starts with:

* An initial state estimate.
* An initial covariance matrix representing the uncertainty in that estimate.

---

# 2. Prediction Step

The prediction step propagates the current state using the motion model.

In my implementation:

* **Position** is predicted using a **constant velocity model**.
* **Heading** is predicted by integrating the **IMU gyroscope angular velocity**.

The predicted state is obtained from the process model:

```text
X(pred) = f(X, u)
```

where:

* `X` is the current state.
* `u` is the control input (IMU gyroscope angular velocity in my implementation).

---

# 3. Covariance Prediction

After predicting the state, the EKF propagates the uncertainty:

```text
P(pred) = F @ P @ Fᵀ + Q
```

where:

* **F** is the Jacobian (state transition matrix) of the motion model.
* **Q** is the process noise covariance.

### State Transition Matrix (F)

`F` describes how uncertainty in the current state propagates to the predicted state through the motion model.

For example, uncertainty in velocity produces uncertainty in the predicted position because position depends on velocity.

### Process Noise Covariance (Q)

`Q` models the uncertainty introduced by the motion model itself.

Examples include:

* Constant velocity assumption not being perfectly true.
* IMU gyroscope measurement noise.
* Unmodeled dynamics such as acceleration.

At the end of the prediction step, the EKF has:

* Predicted state `X(pred)`
* Predicted covariance `P(pred)`

---

# 4. Measurement (Correction) Step

When a GPS measurement is available, the EKF corrects its prediction.

## Measurement Model

The first step is to predict what the GPS should measure from the predicted state:

```text
z(pred) = H @ X(pred)
```

where:

* **H** is the measurement Jacobian.

`H` maps the predicted state into the measurement space by selecting the state variables observed by the sensor.

In my implementation, GPS measures only:

* x position
* y position

---

## Innovation (Residual)

The innovation is the difference between the actual GPS measurement and the predicted GPS measurement:

```text
y = z - z(pred)
```

where:

* `z` is the actual GPS measurement.
* `z(pred)` is the predicted GPS measurement.

The innovation represents the prediction error.

---

## Innovation Covariance

To determine how much confidence to place in the innovation, the EKF computes:

```text
S = H @ P(pred) @ Hᵀ + R
```

where:

* `P(pred)` is the predicted covariance from the prediction step.
* `R` is the measurement noise covariance.

### Measurement Noise Covariance (R)

`R` represents the uncertainty in the GPS measurements.

In my implementation, it is estimated from GPS data by computing the variance of the measured position changes after removing obvious outliers.

---

## Kalman Gain

The Kalman Gain is computed using the innovation covariance.

It determines how much weight should be given to:

* the prediction from the motion model, and

* the GPS measurement.

* Large Kalman Gain → trust the GPS more.

* Small Kalman Gain → trust the prediction more.

---

## State and Covariance Update

The EKF then updates:

* the predicted state, and
* the predicted covariance.

The corrected state and corrected covariance become the starting point for the next EKF iteration.

---

# EKF Flow Summary

```text
Initialize
    │
    ▼
Predict State (Motion Model)
    │
    ▼
Predict Covariance
P(pred) = F P Fᵀ + Q
    │
    ▼
Predict Sensor Measurement
z(pred) = H X(pred)
    │
    ▼
Innovation
y = z - z(pred)
    │
    ▼
Innovation Covariance
S = H P(pred) Hᵀ + R
    │
    ▼
Kalman Gain
    │
    ▼
Correct State
    │
    ▼
Correct Covariance
    │
    ▼
Repeat
```

F matrix → Jacobian of the motion model; describes how uncertainty in the current state propagates to the next predicted state.

Q matrix → Process noise covariance; represents uncertainty introduced by the motion model itself (model assumptions, unmodeled dynamics, IMU noise, etc.).

P matrix → State covariance matrix; represents the uncertainty of the current state estimate.
             After prediction: P_pred represents uncertainty of the predicted state.
             After correction: P_corr represents uncertainty after incorporating measurements.

H matrix → Measurement Jacobian; maps the predicted state into the sensor measurement space and shows which state variables are observed by the sensor.

R matrix → Measurement noise covariance; represents uncertainty in the sensor measurements.
