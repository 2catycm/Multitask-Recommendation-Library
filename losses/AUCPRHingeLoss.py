
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn
import losses.loss_utils as loss_utils
from torch import FloatTensor

class AUCPRHingeLoss(nn.Module):
    """area under the precision-recall curve loss,
    Reference: "Scalable Learning of Non-Decomposable Objectives", Section 5 \
    TensorFlow Implementation: \
    https://github.com/tensorflow/models/tree/master/research/global_objectives\
    """

    def set_config(self, precision_range_lower: float = 0.0, precision_range_upper: float = 1.0,
                   num_classes: int = 1, num_anchors: int = 20, *args, **kwargs):
        """
        Attributes:
            precision_range_lower (float): the lower range of precision values over
                which to compute AUC. Must be nonnegative, `\leq precision_range_upper`,
                and `leq 1.0`.
            precision_range_upper (float): the upper range of precision values over
                which to compute AUC. Must be nonnegative, `\geq precision_range_lower`,
                and `leq 1.0`.
            num_classes (int): number of classes(aka labels)
            num_anchors (int): The number of grid points used to approximate the
                Riemann sum.
        """

        self.precision_range_lower: float = precision_range_lower
        self.precision_range_upper: float = precision_range_upper
        self.num_classes: int = num_classes
        self.num_anchors: int = num_anchors
        
        self.precision_range = (
            self.precision_range_lower,
            self.precision_range_upper,
        )

    def __init__(self, *args, **kwargs):
        """Args:
        args: Config containing `precision_range_lower`, `precision_range_upper`,
            `num_classes`, `num_anchors`
        """
        super().__init__()
        self.set_config(**kwargs)

        # Create precision anchor values and distance between anchors.
        # coresponding to [alpha_t] and [delta_t] in the paper.
        # precision_values: 1D `Tensor` of shape [K], where `K = num_anchors`
        # delta: Scalar (since we use equal distance between anchors)
        self.precision_values, self.delta = loss_utils.range_to_anchors_and_delta(
            self.precision_range, self.num_anchors
        )

        # notation is [b_k] in paper, Parameter of shape [C, K]
        # where `C = number of classes` `K = num_anchors`
        self.biases = nn.Parameter(
            FloatTensor(self.num_classes,
                        self.num_anchors).zero_()
        )
        self.lambdas = nn.Parameter(
            FloatTensor(self.num_classes, self.num_anchors).data.fill_(
                1.0
            )
        )

    def forward(self, logits, targets, reduce=True, size_average=True, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
            size_average (bool, optional): By default, the losses are averaged
                    over observations for each minibatch. However, if the field
                    sizeAverage is set to False, the losses are instead summed
                    for each minibatch. Default: ``True``
            reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per input/target element instead and ignores
                size_average. Default: True
        """
        C = 1 if logits.dim() == 1 else logits.size(1)

        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (
                    self.num_classes, C)
            )

        labels, weights = AUCPRHingeLoss._prepare_labels_weights(
            logits, targets, weights=weights
        )

        # Lagrange multipliers
        # Lagrange multipliers are required to be nonnegative.
        # Their gradient is reversed so that they are maximized
        # (rather than minimized) by the optimizer.
        # 1D `Tensor` of shape [K], where `K = num_anchors`
        lambdas = loss_utils.lagrange_multiplier(self.lambdas)
        # print("lambdas: {}".format(lambdas))

        # A `Tensor` of Shape [N, C, K]
        hinge_loss = loss_utils.weighted_hinge_loss(
            labels.unsqueeze(-1),
            logits.unsqueeze(-1) - self.biases,
            positive_weights=1.0 + lambdas * (1.0 - self.precision_values),
            negative_weights=lambdas * self.precision_values,
        )

        # 1D tensor of shape [C]
        class_priors = loss_utils.build_class_priors(labels, weights=weights)

        # lambda_term: Tensor[C, K]
        # according to paper, lambda_term = lambda * (1 - precision) * |Y^+|
        # where |Y^+| is number of postive examples = N * class_priors
        lambda_term = class_priors.unsqueeze(-1) * (
            lambdas * (1.0 - self.precision_values)
        )

        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term

        # Riemann sum over anchors, and normalized by precision range
        # loss: Tensor[N, C]
        loss = per_anchor_loss.sum(2) * self.delta
        loss /= self.precision_range[1] - self.precision_range[0]

        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()

    @staticmethod
    def _prepare_labels_weights(logits, targets, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
        Returns:
            labels: Tensor of shape [N, C], one-hot representation
            weights: Tensor of shape broadcastable to labels
        """
        N, C = logits.size()
        # Converts targets to one-hot representation. Dim: [N, C]
        labels = FloatTensor(N, C).zero_().scatter(
            1, targets.unsqueeze(1).data, 1)
        # labels = FloatTensor(N, C).zero_().scatter(
        #     1, targets.data, 1)

        if weights is None:
            weights = FloatTensor(N).data.fill_(1.0)

        if weights.dim() == 1:
            weights.unsqueeze_(-1)

        return labels, weights
