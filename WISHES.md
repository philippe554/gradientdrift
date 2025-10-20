
# Wishlist

## Interface

* Command line mode

## Data prep

* ROOT data loader
* Trading sessions
* On-the-fly snapshotting from message data
* Missing value contingency (omit or fill)
* Standard operations (lags, differencing, returns, etc)

## Result output

* TBD

## Test

* The Lagrange Multiplier (LM) Test

## Internals

* Numerical stability test (nan / inf)
* Gradient clipping
* Checkpoints
* Logging
* Panel data: Clustered Standard Errors
* Instrumental Variables (IV) / Two-Stage Least Squares (2SLS)

## Future plans

* Gaussian Mixture Model (GMM)
* Bayesian Models

## Formula Syntax

### Non timeseries basics

Basic Regression & Transformations

```
y ~ x1 + x2
log(y) ~ x1 + exp(x2)
```

Multivariate Endogenous Variables (for VAR-style models)

```
y1, y2 ~ x1 + x2
```

### Time series components

Explicit Lag and Residual Functions

```
y ~ lag(y, [1, 2, 5]) + lag(x1, 0)
y ~ res(y, [1, 2])
y1, y2 ~ lag(y1, 1) + lag(y2, [1, 2]) + res(y1, 1)
```

High-Level Macros for Symmetric Models

```
y1, y2 ~ AR(1:2) + MA(1) + x1
y1, y2, y3 ~ AR(2) + MA([1, 4]) + lag(x1, 0:2)
```

ARIMA Template Example

```
ARIMA(p=2, d=1, q=1, endog=['y1', 'y2'], exog=['x1'])
-->
diff(y1), diff(y2) ~ AR(1:2) + MA(1) + x1
```

Differencing Variations

```
diff(y, order=2)
diff(y, lag=12)
diff(diff(y), lag=12)
```

Random effect

```
# Fixed effect:
test_score ~ {intercept} + {beta_study_hours} * study_hours + {alpha[school_id]}

# Mixed effect:
{random_effect[school_id]} ~ normal(0, {sigma_school_effect})
test_score ~ {intercept} + {beta_study_hours} * study_hours + {random_effect[school_id]}

# Mixed effect including slope:
{u_intercept[school_id]}, {u_slope[school_id]} ~ mvnormal([0, 0], TBD)
test_score ~ {intercept} + {beta_study_hours} * study_hours + {u_intercept[school_id]} + study_hours * {u_slope[school_id]}
```

### Alternative loss functions

Logistic Regression and alternatives

```
logistic(y) ~ x1 + x2
poisson(y) ~ x1 + x2
```

GARCH example: custom loss function and state space models

```
0 < {alpha},{beta},{omega}
sigmaSq[0] = {omega} / (1 - {alpha} - {beta})
sigmaSq = omega + alpha * (lag(y) - {mu}) ** 2 + beta * lag(sigmaSq)
maximize: norm.logpdf(y, {mu}, sqrt(sigmaSq))
```

Summary:

Lines with an `~` are models where the loss function is automatically configured based on the LHS.
Lines with an `=` are simple assignments, and can be used with custom loss functions or used as hidden state variables.

### Parameter constraints

This is not a constrainted optimizer, these constraints are simply enforced using reparameterization. For now, only upper and lower bounds for individual coefficients are allowed.

```
{W1} = [128, 64] = 4
{W1},{W2} = [4, 4] ~ N(0, 0.1)
0 < {W1} < 1
sum({W1}) == 4
sum({W1},{W2}) == 5
```

# Latent Variable models

Solvers:

MCMC (first RWMH, later NUTS and SGMCMC)
    Metropolis-Adjusted Langevin Algorithm (MALA) (include gradient)
VI
EM with Kalman smoother for the E step
EM with VI for the E step "Variational EM (VEM)"
EM with MCMC for the E step

# Tests

```
    CointegrationModel:
        y2 ~ {beta} * y1 + {mu}

    ADFTest:
        ecm = lag(y2) - {CointegrationModel.beta} * lag(y1) - {CointegrationModel.mu}
        diff(ecm) ~ {mu} + {gamma} * lag(ecm) + {beta} * lag(diff(ecm))
        tests:
            ttest({mu})
            # Or explicitly:
            tstat({mu}) ~ TDist(DF = 3) | test(H0=0, alternative='two-sided')

            ADFTest({gamma})
            # Or explicitly:
            tstat({gamma}) ~ DFDist() | test(H0=0, alternative='left-sided')
        
    VECM:
        ecm = lag(y2) - {CointegrationModel.beta} * lag(y1) - {CointegrationModel.mu}

        diff(y1) ~ {mu1} + {alpha1} * ecm + {gamma11} * lag(diff(y1)) + {gamma12} * lag(diff(y2))
        diff(y2) ~ {mu2} + {alpha2} * ecm + {gamma21} * lag(diff(y1)) + {gamma22} * lag(diff(y2))

        tests:
            residuals(y1) ~ normal(0, {sigma1}) | KS()
            {alpha1} ~ Bootstrap(reps=10000) | test(H0=0, alternative='two-sided')

    ANOVA:
        # <model spec>
        tests:
            var_between(y) / var_within(y) ~ FDist() | test(H0=1, alternative='two-sided')

    UnrestrictedModel:
        y ~ {mu} + {beta1} * x1 + {beta2} * x2 + {beta3} * x3

    RestrictedModel:
        y ~ {mu} + {beta1} * x1
    
    tests:
        FStat(UnrestrictedModel, RestrictedModel) ~ FDist() | test(H0=0, alternative='two-sided')

```