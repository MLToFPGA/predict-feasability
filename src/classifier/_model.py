from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import rustuna
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from xgboost.callback import TrainingCallback

Metric = Callable[..., Any]
Objective = Callable[[Any, Any], tuple[np.ndarray, np.ndarray]]


def create_model(config: XGBClassifierConfig | None = None) -> xgb.XGBClassifier:
    """Build an :class:`xgboost.XGBClassifier` from ``config``.

    A ``None`` config uses XGBoost defaults throughout.
    """
    if config is None:
        config = XGBClassifierConfig()
    return xgb.XGBClassifier(**config.to_params())


class Fitter:
    def __init__(
        self,
        search_space: XGBClassifierSearchSpace | None = None,
        n_trials: int = 100,
    ):
        self._search_space = search_space or XGBClassifierSearchSpace.default()
        self._model: xgb.XGBClassifier | None = None
        self._n_trials = n_trials

    def fit(
        self,
        x: NDArray[Any],
        y: NDArray[Any],
        x_val: NDArray[Any],
        y_val: NDArray[Any],
    ) -> Self:
        def objective(trial: rustuna.Trial) -> float:
            config = self._search_space.suggest(trial)
            model = create_model(config)
            model.fit(x, y, eval_set=[(x_val, y_val)], verbose=False)
            proba = model.predict_proba(x_val)[:, 1]
            return float(roc_auc_score(y_val, proba))

        study = rustuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self._n_trials)

        best = self._search_space.config_from_params(study.best_trial.params)
        self._model = create_model(best)
        self._model.fit(x, y)
        return self

    def get_model(self) -> xgb.XGBClassifier:
        if not self._model:
            raise
        return self._model


@dataclass(slots=True, frozen=True)
class XGBClassifierConfig:
    """Typed configuration for :class:`xgboost.XGBClassifier`.

    Every field mirrors a constructor argument of ``XGBClassifier``. Fields left
    as ``None`` are dropped when the model is built, so XGBoost falls back to its
    own default for them. Pass an instance to :func:`create_model`.
    """

    # --- Core boosting parameters ---------------------------------------
    n_estimators: int | None = None
    """Number of boosting rounds."""

    max_depth: int | None = None
    """Maximum tree depth for base learners."""

    max_leaves: int | None = None
    """Maximum number of leaves; 0 indicates no limit."""

    max_bin: int | None = None
    """If using histogram-based algorithm, maximum number of bins per feature."""

    grow_policy: str | None = None
    """Tree growing policy: ``"depthwise"`` or ``"lossguide"``."""

    learning_rate: float | None = None
    """Boosting learning rate (xgb's "eta")."""

    verbosity: int | None = None
    """Degree of verbosity, 0 (silent) - 3 (debug)."""

    objective: str | Objective | None = None
    """Learning task / objective, or a custom objective callable."""

    booster: str | None = None
    """Which booster to use: ``"gbtree"``, ``"gblinear"`` or ``"dart"``."""

    tree_method: str | None = None
    """Tree method to use. Defaults to ``"auto"``."""

    n_jobs: int | None = None
    """Number of parallel threads used to run xgboost."""

    # --- Regularization / tree shape ----------------------------------
    gamma: float | None = None
    """(min_split_loss) Minimum loss reduction to make a further partition."""

    min_child_weight: float | None = None
    """Minimum sum of instance weight (hessian) needed in a child."""

    max_delta_step: float | None = None
    """Maximum delta step allowed for each tree's weight estimation."""

    subsample: float | None = None
    """Subsample ratio of the training instances."""

    sampling_method: str | None = None
    """Sampling method (GPU hist only): ``"uniform"`` or ``"gradient_based"``."""

    colsample_bytree: float | None = None
    """Subsample ratio of columns when constructing each tree."""

    colsample_bylevel: float | None = None
    """Subsample ratio of columns for each level."""

    colsample_bynode: float | None = None
    """Subsample ratio of columns for each split."""

    reg_alpha: float | None = None
    """L1 regularization term on weights (xgb's alpha)."""

    reg_lambda: float | None = None
    """L2 regularization term on weights (xgb's lambda)."""

    scale_pos_weight: float | None = None
    """Balancing of positive and negative weights."""

    base_score: float | list[float] | None = None
    """Initial prediction score of all instances, global bias."""

    # --- Misc -------------------------------------------------------
    random_state: np.random.RandomState | np.random.Generator | int | None = None
    """Random number seed."""

    missing: float = float("nan")
    """Value in the data which needs to be treated as missing."""

    num_parallel_tree: int | None = None
    """Used for boosting random forest."""

    monotone_constraints: dict[str, int] | str | None = None
    """Constraint of variable monotonicity."""

    interaction_constraints: str | list[tuple[str, ...]] | None = None
    """Constraints for permitted feature interactions."""

    importance_type: str | None = None
    """Feature importance type for ``feature_importances_``."""

    device: str | None = None
    """Device ordinal: ``"cpu"``, ``"cuda"`` or ``"gpu"``."""

    validate_parameters: bool | None = None
    """Give warnings for unknown parameters."""

    enable_categorical: bool = False
    """Enable categorical feature support (see ``DMatrix``)."""

    feature_types: Sequence[str] | None = None
    """Feature types, for specifying without constructing a dataframe."""

    feature_weights: Any | None = None
    """Per-feature weight for column sampling. All values must be > 0."""

    max_cat_to_onehot: int | None = None
    """Threshold for one-hot based split for categorical data (experimental)."""

    max_cat_threshold: int | None = None
    """Max number of categories considered for each split (experimental)."""

    multi_strategy: str | None = None
    """Multi-target training strategy: ``"one_output_per_tree"`` or
    ``"multi_output_tree"`` (work in progress)."""

    # --- Training / early stopping ---------------------------------
    eval_metric: str | list[str | Metric] | Metric | None = None
    """Metric(s) used for monitoring training and early stopping."""

    early_stopping_rounds: int | None = None
    """Activates early stopping. Requires at least one item in ``eval_set``."""

    callbacks: list[TrainingCallback] | None = None
    """Callback functions applied at the end of each iteration."""

    # --- Extra Booster kwargs -------------------------------------
    kwargs: dict[str, Any] = field(default_factory=dict[str, Any])
    """Extra keyword arguments forwarded to the XGBoost Booster."""

    def to_params(self) -> dict[str, Any]:
        """Return constructor kwargs, dropping unset (``None``) fields.

        ``missing`` and ``enable_categorical`` always keep their explicit
        defaults; ``kwargs`` is flattened into the result.
        """
        always_keep = {"missing", "enable_categorical"}
        params: dict[str, Any] = {}
        for name, value in asdict(self).items():
            if name == "kwargs":
                continue
            if value is None and name not in always_keep:
                continue
            params[name] = value
        params.update(self.kwargs)
        return params


@runtime_checkable
class Trial(Protocol):
    """The slice of the optuna / rustuna ``Trial`` API the spaces below need."""

    def suggest_int(
        self, name: str, low: int, high: int, step: int = ..., log: bool = ...
    ) -> int: ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        step: float | None = ...,
        log: bool = ...,
    ) -> float: ...

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any: ...


@runtime_checkable
class ParamSpace(Protocol):
    """A search space for a single hyper-parameter.

    ``suggest`` receives the running ``trial`` and the parameter name and must
    return a concrete value, normally by delegating to ``trial.suggest_*``.
    """

    def suggest(self, trial: Trial, name: str) -> Any: ...


@dataclass(slots=True, frozen=True)
class IntSpace:
    """Integer range sampled with ``trial.suggest_int``."""

    low: int
    high: int
    step: int = 1
    log: bool = False

    def suggest(self, trial: Trial, name: str) -> int:
        return trial.suggest_int(
            name, self.low, self.high, step=self.step, log=self.log
        )


@dataclass(slots=True, frozen=True)
class FloatSpace:
    """Float range sampled with ``trial.suggest_float``."""

    low: float
    high: float
    step: float | None = None
    log: bool = False

    def suggest(self, trial: Trial, name: str) -> float:
        return trial.suggest_float(
            name, self.low, self.high, step=self.step, log=self.log
        )


@dataclass(slots=True, frozen=True)
class CategoricalSpace:
    """Fixed set of choices sampled with ``trial.suggest_categorical``."""

    choices: tuple[Any, ...]

    def __init__(self, choices: Sequence[Any]) -> None:
        object.__setattr__(self, "choices", tuple(choices))

    def suggest(self, trial: Trial, name: str) -> Any:
        return trial.suggest_categorical(name, list(self.choices))


@dataclass(slots=True, frozen=True)
class Fixed:
    """A constant — no ``trial`` call, the value is used as-is."""

    value: Any

    def suggest(self, trial: Trial, name: str) -> Any:
        return self.value


@dataclass(slots=True, frozen=True)
class XGBClassifierSearchSpace:
    """Search space mirroring the tunable fields of :class:`XGBClassifierConfig`.

    Each field is a :class:`ParamSpace` (``IntSpace`` / ``FloatSpace`` /
    ``CategoricalSpace`` / ``Fixed``) or ``None``. :meth:`suggest` walks the set
    fields, samples each one from the trial and returns a fully-formed
    :class:`XGBClassifierConfig`::

        space = XGBClassifierSearchSpace.default()


        def objective(trial: rustuna.Trial) -> float:
            config = space.suggest(trial)
            model = create_model(config)
            ...
    """

    n_estimators: ParamSpace | None = None
    max_depth: ParamSpace | None = None
    max_leaves: ParamSpace | None = None
    max_bin: ParamSpace | None = None
    grow_policy: ParamSpace | None = None
    learning_rate: ParamSpace | None = None
    booster: ParamSpace | None = None
    tree_method: ParamSpace | None = None
    gamma: ParamSpace | None = None
    min_child_weight: ParamSpace | None = None
    max_delta_step: ParamSpace | None = None
    subsample: ParamSpace | None = None
    sampling_method: ParamSpace | None = None
    colsample_bytree: ParamSpace | None = None
    colsample_bylevel: ParamSpace | None = None
    colsample_bynode: ParamSpace | None = None
    reg_alpha: ParamSpace | None = None
    reg_lambda: ParamSpace | None = None
    scale_pos_weight: ParamSpace | None = None
    base_score: ParamSpace | None = None
    num_parallel_tree: ParamSpace | None = None
    max_cat_to_onehot: ParamSpace | None = None
    max_cat_threshold: ParamSpace | None = None
    multi_strategy: ParamSpace | None = None

    #: Extra Booster params keyed by name, forwarded through ``config.kwargs``.
    kwargs: Mapping[str, ParamSpace] = field(default_factory=dict[str, ParamSpace])

    def suggest(self, trial: Trial) -> XGBClassifierConfig:
        """Sample every set field from ``trial`` and build a config."""
        params: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "kwargs":
                continue
            space: ParamSpace | None = getattr(self, f.name)
            if space is not None:
                params[f.name] = space.suggest(trial, f.name)
        extra = {name: sp.suggest(trial, name) for name, sp in self.kwargs.items()}
        if extra:
            params["kwargs"] = extra
        return XGBClassifierConfig(**params)

    def config_from_params(self, params: Mapping[str, Any]) -> XGBClassifierConfig:
        """Rebuild a config from a flat ``{name: value}`` mapping.

        Use with a completed study, e.g.
        ``space.config_from_params(study.best_trial.params)``. Names registered
        under :attr:`kwargs` are routed back into ``config.kwargs``. ``Fixed``
        spaces are re-applied from the space itself, since their values never
        reach the trial and so are absent from ``params``.
        """
        init: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for f in fields(self):
            if f.name == "kwargs":
                continue
            space: ParamSpace | None = getattr(self, f.name)
            if isinstance(space, Fixed):
                init[f.name] = space.value

        for name, value in params.items():
            if name in self.kwargs:
                extra[name] = value
            else:
                init[name] = value

        for name, sp in self.kwargs.items():
            if isinstance(sp, Fixed):
                extra[name] = sp.value

        if extra:
            init["kwargs"] = extra
        return XGBClassifierConfig(**init)

    @classmethod
    def default(cls) -> Self:
        """A reasonable starting search space for binary classification.

        Tune the ranges to your data — this is a sane default, not a rule.
        """
        return cls(
            n_estimators=IntSpace(100, 2000, step=100),
            learning_rate=FloatSpace(1e-3, 0.3, log=True),
            max_depth=IntSpace(2, 12),
            min_child_weight=FloatSpace(1e-1, 20.0, log=True),
            gamma=FloatSpace(0.0, 5.0),
            subsample=FloatSpace(0.5, 1.0),
            colsample_bytree=FloatSpace(0.5, 1.0),
            colsample_bylevel=FloatSpace(0.5, 1.0),
            reg_alpha=FloatSpace(1e-8, 10.0, log=True),
            reg_lambda=FloatSpace(1e-8, 10.0, log=True),
            grow_policy=CategoricalSpace(("depthwise", "lossguide")),
        )
