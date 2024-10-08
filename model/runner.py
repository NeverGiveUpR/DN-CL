import json
import os

import aclick
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from sympy import sympify

from symformer.dataset.tokenizers import GeneralTermTokenizer
from symformer.dataset.utils import generate_all_possible_ranges
from symformer.dataset.utils.sympy_functions import evaluate_points, expr_to_func

from symformer.utils import pull_model
from .config import Config
from .model import TransformerWithRegressionAsSeq
from .utils.const_improver import OptimizationType
from .utils.convertor import BestFittingFilter
from .utils.decoding import RandomSampler, TopK


def sample_points(func, num_vars):
    for left, right in generate_all_possible_ranges(num_vars, -5, 5):
        try:
            x = np.random.uniform(left, right, (100 * num_vars, num_vars))
            y = evaluate_points(func, x)

            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                raise RuntimeWarning("Is nan or inf!")

            if not np.all(np.isfinite(y)):
                raise RuntimeWarning("Not finite")

            res = np.concatenate((x, np.reshape(y, (-1, 1))), axis=1)
            return res
        except RuntimeWarning:
            continue

    raise RuntimeError("No range found")

class Runner:
    def __init__(
        self,
        model_path,
        config,
        variables,
        sampler=TopK(20),
        optimization_type: OptimizationType = "gradient",
        use_pool=True,
        early_stopping=False,
        num_equations=256,
    ):
        self.config = config
        self.variables = variables
        self.tokenizer = GeneralTermTokenizer(variables)
        self.model = self.load_model(model_path, config)
        self.search = RandomSampler(
            sampler,
            50,
            self.tokenizer,
            self.model,
            True,
            BestFittingFilter(
                self.tokenizer, optimization_type, use_pool, early_stopping
            ),
            num_equations,
        )

    @classmethod
    def from_checkpoint(cls, path, types='weights', **kwargs):
        path = pull_model(path)
        with open(os.path.join(path, "config.json")) as f:
            config_dict = json.load(f)
        config_dict["dataset_config"]["path"] = ""
        config_dict["dataset_config"]["valid_path"] = ""
        config: Config = aclick.utils.from_dict(Config, config_dict)
        if types=='weights':
            return cls(
                os.path.join(path, "weights"),
                config,
                variables=config.dataset_config.variables,
                **kwargs
            )
        else:
            return cls(
                os.path.join(path, "last"),
                config,
                variables=config.dataset_config.variables,
                **kwargs
            )

    def load_model(self, model_path, config):
        strategy = tf.distribute.get_strategy()
        transformer = TransformerWithRegressionAsSeq(
            config, self.tokenizer, config.input_regularizer, strategy
        )
        transformer.build(
            input_shape=[
                (
                    None,
                    config.dataset_config.num_points,
                    len(config.dataset_config.variables) + 1,
                ),
                (None, None),
                (None, None),
            ]
        )
        transformer.compile()
        transformer.load_weights(model_path)
        return transformer
    
    def predict(self, equation, clean_points=None, noise_points=None, noise=None, opt_const_by_clean_points=True, evaluate_with_noise=False):
        # sym_eq = sympify(equation)
        # lam = expr_to_func(sym_eq, self.variables)
        # if clean_points is None:
        #     clean_points = sample_points(lam, len(self.variables))
        # if noise_points is None:
        #     noise_points = adding_gaussian_noise(clean_points, noise_std=noise)
            
        clean_points = tf.convert_to_tensor([clean_points])
        noise_points = tf.convert_to_tensor([noise_points])

        # this function could choose whether optimizing constant by clean points 
        prediction = self.search.batch_decode(noise_points, clean_points,
                                              use_clean_points_opt_const=opt_const_by_clean_points)
        pred_inorder = prediction[-2][0]
        prediction = prediction[-1][0]
        golden_lambda = expr_to_func(prediction, self.variables)

        if evaluate_with_noise:
            noise_points = noise_points.numpy().squeeze()
            pred_y = evaluate_points(
                golden_lambda, tf.reshape(noise_points[:, :-1], (-1, len(self.variables)))
            )
            pred_y = np.nan_to_num(pred_y, nan=1000000, posinf=1000000, neginf=1000000)
            return (
                pred_inorder,
                r2_score(noise_points[:, -1], pred_y.reshape([-1])),
                np.mean(
                    np.abs(noise_points[:,-1]-pred_y.reshape([-1])) / np.abs(noise_points[:, -1])
                )
            )
        else:
            clean_points = clean_points.numpy().squeeze()
            pred_y = evaluate_points(
                golden_lambda, tf.reshape(clean_points[:, :-1], (-1, len(self.variables)))
            )
            pred_y = np.nan_to_num(pred_y, nan=1000000, posinf=1000000, neginf=1000000)
            return (
                pred_inorder,
                r2_score(clean_points[:, -1], pred_y.reshape([-1])),
                np.mean(
                    np.abs(clean_points[:, -1] - pred_y.reshape([-1])) / np.abs(clean_points[:, -1])
                ),
            )


    # def predict(self, equation, clean_points=None, noise_points=None, noise=None, evaluate_with_noise=False):
    #     sym_eq = sympify(equation)
    #     lam = expr_to_func(sym_eq, self.variables)
    #     if points is None:
    #         points = sample_points(lam, len(self.variables))
    #         points = tf.convert_to_tensor([points])

    #     if noise is not None:
    #         points = adding_gaussian_noise(points, noise_std=noise)
    #         print("noise level {}".format(str(noise)))

    #     prediction = self.search.batch_decode(points)
    #     pred_inorder = prediction[-2][0]
    #     prediction = prediction[-1][0]
    #     golden_lambda = expr_to_func(prediction, self.variables)

    #     points = sample_points(lam, len(self.variables))
    #     pred_y = evaluate_points(
    #         golden_lambda, tf.reshape(points[:, :-1], (-1, len(self.variables)))
    #     )
    #     return (
    #         pred_inorder,
    #         r2_score(points[:, -1], pred_y.reshape([-1])),
    #         np.mean(
    #             np.abs(points[:, -1] - pred_y.reshape([-1])) / np.abs(points[:, -1])
    #         ),
    #     )

    def predict_all(self, equation=None, points=None):
        if points is None:
            assert equation is not None
            sym_eq = sympify(equation)
            lam = expr_to_func(sym_eq, self.variables)
            if points is None:
                points = sample_points(lam, len(self.variables))
                points = tf.convert_to_tensor([points])

        prediction = self.search.batch_decode(points, return_all=True)
        return prediction
    
    def predict_feature(self, points=None):
        points = tf.convert_to_tensor([points])
        features = self.search.encoder_feature(points)
        return features.numpy()